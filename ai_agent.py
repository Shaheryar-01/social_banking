import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import jsonschema
import re
from prompts import (
    filter_extraction_prompt,
    pipeline_generation_prompt,
    response_prompt,
    query_prompt,
    intent_prompt,
    transfer_prompt
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LangChain LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1
)

# MongoDB pipeline schema for validation
PIPELINE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "$match": {"type": "object"},
            "$group": {"type": "object"},
            "$sort": {"type": "object"},
            "$limit": {"type": "integer", "minimum": 1},
            "$project": {"type": "object"}
        },
        "additionalProperties": False
    }
}

class FilterExtraction(BaseModel):
    description: Optional[str] = None
    category: Optional[str] = None
    month: Optional[str] = None
    year: Optional[int] = None
    transaction_type: Optional[str] = None
    amount_range: Optional[Dict[str, float]] = None
    date_range: Optional[Dict[str, str]] = None
    limit: Optional[int] = None

class QueryResult(BaseModel):
    intent: str = Field(default="general")
    pipeline: List[Dict[str, Any]] = Field(default_factory=list)
    response_format: str = Field(default="natural_language")
    filters: Optional[FilterExtraction] = None

def month_to_number(month: str) -> int:
        """Convert month name to number."""
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
        }
        return months.get(month.lower(), 1)

def month_days(month: str, year: int) -> int:
    """Get number of days in a month."""
    month_num = month_to_number(month)
    if month_num in [4, 6, 9, 11]:
        return 30
    elif month_num == 2:
        return 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    else:
        return 31
        
_BRACE_RE = re.compile(r'[{[]')

def _find_json_span(text: str) -> Tuple[int, int]:
    """Return (start, end) indices of the first JSON value in text."""
    m = _BRACE_RE.search(text)
    if not m:
        raise ValueError("No '{' or '[' found")
    start = m.start()
    stack = [text[start]]
    for i in range(start + 1, len(text)):
        ch = text[i]
        if ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if not stack:
                break
            open_ch = stack.pop()
            # naive but fine for LLM output
            if not stack:
                return start, i + 1
    raise ValueError("Unbalanced brackets")

def _json_fix(raw: str) -> str:
    """Best‑effort clean‑ups that keep strict JSON subset."""
    fixed = raw.strip()

    # ``` fences or other wrappers already removed by caller
    fixed = re.sub(r"'", '"', fixed)                     # single → double quotes
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)         # trailing comma
    fixed = fixed.replace('NaN', 'null')                 # NaN → null
    fixed = fixed.replace('Infinity', '1e308')           # Infinity → big number
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', fixed)  # stray backslashes
    return fixed

    
class BankingAIAgent:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        
    def extract_json_from_response(self, raw: str) -> Optional[Any]:
        """
        Extract the first JSON value from an LLM reply.
        Returns the parsed Python object or None.
        """
        logger.info({"action": "extract_json_start", "raw_sample": raw[:200]})

        try:
            start, end = _find_json_span(raw)
            candidate = raw[start:end]
        except ValueError as e:
            logger.error({"action": "extract_json_span_fail", "error": str(e)})
            return None

        # quick parse
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through

        # repair & retry
        candidate = _json_fix(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.error({"action": "extract_json_parse_fail", "error": str(e), "candidate": candidate[:200]})
            return None
    
    def extract_filters_with_llm(self, user_message: str) -> FilterExtraction:
        """Use LLM to extract filters from user query."""
        try:
            logger.info({
                "action": "extract_filters_with_llm",
                "user_message": user_message
            })
            
            response = llm.invoke([
                SystemMessage(content=filter_extraction_prompt.format(
                    user_message=user_message,
                    current_date=datetime.now().strftime("%Y-%m-%d")
                ))
            ])
            
            logger.info({
                "action": "llm_filter_extraction_response",
                "response_content": response.content
            })
            
            try:
                filters_obj = self.extract_json_from_response(response.content)
                if filters_obj is None:
                    raise ValueError("Could not parse filter JSON")
                filters = FilterExtraction(**filters_obj)

                logger.info({
                    "action": "filters_extracted",
                    "filters": filters.dict()
                })
                return filters
            
            except (json.JSONDecodeError, TypeError) as e:
                logger.error({
                    "action": "filter_extraction_parse_error",
                    "error": str(e),
                    "raw_response": response.content
                })
                return FilterExtraction()
                
        except Exception as e:
            logger.error({
                "action": "extract_filters_with_llm",
                "error": str(e)
            })
            return FilterExtraction()
        
        
    def generate_pipeline_from_filters(self, filters: FilterExtraction, intent: str, account_number: str) -> List[Dict[str, Any]]:
        """Generate MongoDB pipeline from extracted filters using LLM."""
        try:
            logger.info({
                "action": "generate_pipeline_from_filters",
                "filters": filters.dict(),
                "intent": intent,
                "account_number": account_number
            })
            
            response = llm.invoke([
                SystemMessage(content=pipeline_generation_prompt.format(
                    filters=json.dumps(filters.dict()),
                    intent=intent,
                    account_number=account_number
                ))
            ])
            
            logger.info({
                "action": "llm_pipeline_generation_response",
                "response_content": response.content[:1000]  # Log first 1000 chars for debugging
            })
            
            cleaned_response = self.extract_json_from_response(response.content)
        
            if not cleaned_response:
                logger.error({
                    "action": "pipeline_generation_no_json_found",
                    "raw_response": response.content[:1000]
                })
                return self._generate_fallback_pipeline(filters, intent, account_number)
            
            # Since response is already a list, no need to parse again
            pipeline = cleaned_response
            
            # Validate pipeline structure
            jsonschema.validate(pipeline, PIPELINE_SCHEMA)
            logger.info({
                "action": "pipeline_generated",
                "pipeline": pipeline
            })
            return pipeline
        except json.JSONDecodeError as e:
            logger.error({
                "action": "pipeline_generation_parse_error",
                "error": str(e),
                "cleaned_response": cleaned_response[:1000]
            })
            return self._generate_fallback_pipeline(filters, intent, account_number)
            
        except jsonschema.ValidationError as e:
            logger.error({
                "action": "pipeline_validation_error",
                "error": str(e),
                "cleaned_response": cleaned_response[:1000]
            })
            return self._generate_fallback_pipeline(filters, intent, account_number)
                    
        except Exception as e:
            logger.error({
                "action": "generate_pipeline_from_filters",
                "error": str(e)
            })
            return self._generate_fallback_pipeline(filters, intent, account_number)
    
    def _generate_fallback_pipeline(self, filters: FilterExtraction, intent: str, account_number: str) -> List[Dict[str, Any]]:
        """Generate a basic pipeline when LLM fails."""
        logger.info({
            "action": "generating_fallback_pipeline",
            "intent": intent,
            "filters": filters.dict()
        })
        
        # Basic match stage
        match_stage = {"$match": {"account_number": account_number}}
        
        # For transaction history
        if intent == "transaction_history":
            pipeline = [
                match_stage,
                {"$sort": {"date": -1, "_id": -1}}
            ]
            if filters.limit:
                pipeline.append({"$limit": filters.limit})
            return pipeline
        
        # For spending analysis or category spending
        elif intent in ["spending_analysis", "category_spending"]:
            if filters.transaction_type:
                match_stage["$match"]["type"] = filters.transaction_type
            
            if filters.description:
                match_stage["$match"]["description"] = {
                    "$regex": filters.description,
                    "$options": "i"
                }
            
            if filters.category:
                match_stage["$match"]["category"] = {
                    "$regex": filters.category,
                    "$options": "i"
                }
            
            # Add date range if month/year specified
            if filters.month and filters.year:
                month_num = month_to_number(filters.month)
                days_in_month = month_days(filters.month, filters.year)
                match_stage["$match"]["date"] = {
                    "$gte": {"$date": f"{filters.year}-{month_num:02d}-01T00:00:00Z"},
                    "$lte": {"$date": f"{filters.year}-{month_num:02d}-{days_in_month:02d}T23:59:59Z"}
                }
            
            pipeline = [
                match_stage,
                {
                    "$group": {
                        "_id": None,
                        "total_usd": {"$sum": "$amount_usd"},
                        "total_pkr": {"$sum": "$amount_pkr"}
                    }
                }
            ]
            return pipeline
        
        # Default fallback
        return [match_stage, {"$sort": {"date": -1, "_id": -1}}, {"$limit": 10}]

    def detect_intent_from_filters(self, user_message: str, filters: FilterExtraction) -> str:
        """Detect intent using LLM for more flexible understanding."""
        try:
            logger.info({
                "action": "llm_intent_classification",
                "user_message": user_message,
                "filters": filters.dict()
            })
            
            # Call LLM for intent classification
            response = llm.invoke([
                SystemMessage(content=intent_prompt.format(
                    user_message=user_message,
                    filters=json.dumps(filters.dict())
                ))
            ])
            
            # Clean and validate response
            detected_intent = response.content.strip().lower()
            
            # Valid intents
            valid_intents = [
                "balance_inquiry",
                "transaction_history", 
                "spending_analysis",
                "category_spending",
                "transfer_money",
                "general"
            ]
            
            # Check if returned intent is valid
            if detected_intent in valid_intents:
                logger.info({
                    "action": "intent_detected",
                    "intent": detected_intent,
                    "user_message": user_message
                })
                return detected_intent
            else:
                # Fallback: try to match partial responses
                for intent in valid_intents:
                    if intent in detected_intent:
                        logger.info({
                            "action": "intent_detected_partial",
                            "intent": intent,
                            "raw_response": detected_intent,
                            "user_message": user_message
                        })
                        return intent
                
                # If no match found, return general
                logger.warning({
                    "action": "intent_detection_failed",
                    "raw_response": detected_intent,
                    "user_message": user_message,
                    "fallback_intent": "general"
                })
                return "general"
                
        except Exception as e:
            logger.error({
                "action": "llm_intent_classification",
                "error": str(e),
                "user_message": user_message
            })
            
            # Fallback to rule-based approach if LLM fails
            logger.info({
                "action": "fallback_to_rule_based",
                "user_message": user_message
            })
            return self._rule_based_intent_fallback(user_message, filters)

    def _rule_based_intent_fallback(self, user_message: str, filters: FilterExtraction) -> str:
        """Fallback rule-based intent detection with expanded keyword matching."""
        user_message_lower = user_message.lower()
        
        # Balance inquiry keywords
        balance_keywords = ["balance", "money", "amount", "funds", "account", "cash"]
        
        # Transaction history keywords  
        transaction_keywords = ["transaction", "history", "recent", "last", "show", "list", "activities"]
        
        # Spending analysis keywords
        spending_keywords = ["spend", "spent", "spending", "expenditure", "expense", "expenses", 
                            "cost", "costs", "paid", "pay", "payment", "purchase", "purchased", 
                            "buying", "bought", "money went", "charged"]
        
        # Transfer keywords
        transfer_keywords = ["transfer", "send", "pay", "wire", "remit", "move money"]
        
        # Check for balance inquiry
        if any(keyword in user_message_lower for keyword in balance_keywords):
            return "balance_inquiry"
        
        # Check for transaction history
        elif any(keyword in user_message_lower for keyword in transaction_keywords) or filters.limit:
            return "transaction_history"
        
        # Check for spending analysis
        elif any(keyword in user_message_lower for keyword in spending_keywords):
            if filters.category:
                return "category_spending"
            else:
                return "spending_analysis"
        
        # Check for transfer
        elif any(keyword in user_message_lower for keyword in transfer_keywords):
            return "transfer_money"
        
        else:
            return "general"

    def detect_intent_fallback(self, user_message: str) -> tuple[str, List[Dict[str, Any]]]:
        """Improved fallback intent detection using LLM filter extraction."""
        logger.info({
            "action": "detect_intent_fallback",
            "user_message": user_message
        })
        
        # Extract filters using LLM
        filters = self.extract_filters_with_llm(user_message)
        
        # Detect intent from filters
        intent = self.detect_intent_from_filters(user_message, filters)
        
        # Generate pipeline from filters
        pipeline = self.generate_pipeline_from_filters(filters, intent, "{{account_number}}")
        
        logger.info({
            "action": "fallback_intent_result",
            "intent": intent,
            "filters": filters.dict(),
            "pipeline": pipeline
        })
        
        return intent, pipeline

    def replace_account_number_in_pipeline(self, pipeline: List[Dict[str, Any]], account_number: str) -> List[Dict[str, Any]]:
        """Recursively replace {{account_number}} placeholder in pipeline."""
        def replace_in_dict(obj):
            if isinstance(obj, dict):
                return {k: replace_in_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_dict(item) for item in obj]
            elif isinstance(obj, str):
                return obj.replace("{{account_number}}", account_number)
            else:
                return obj
        
        return replace_in_dict(pipeline)

    async def process_query(self, user_message: str, account_number: str, first_name: str) -> str:
        """Process user banking queries using LLM and dynamic MongoDB pipeline."""
        logger.info({
            "action": "process_query",
            "user_message": user_message,
            "account_number": account_number,
            "first_name": first_name
        })

        # Step 1: Analyze query and generate pipeline
        query_analysis = await self._analyze_query(user_message, account_number)
        logger.info({
            "action": "query_analysis_result",
            "query_analysis": query_analysis.dict(),
            "user_message": user_message
        })

        # Step 2: Validate pipeline
        if query_analysis.pipeline:
            try:
                jsonschema.validate(query_analysis.pipeline, PIPELINE_SCHEMA)
                logger.info({"action": "pipeline_validation", "status": "success"})
            except jsonschema.ValidationError as e:
                logger.error({
                    "action": "pipeline_validation",
                    "status": "failed",
                    "error": str(e)
                })
                return "Error: Invalid MongoDB pipeline generated. Please try rephrasing your query."

        # Step 3: Execute appropriate action
        if query_analysis.intent == "balance_inquiry":
            logger.info({"action": "execute_balance_inquiry"})
            return await self._handle_balance_inquiry(account_number, first_name, query_analysis, user_message)
        elif query_analysis.intent in ["transaction_history", "spending_analysis", "category_spending"]:
            logger.info({"action": "execute_data_query", "intent": query_analysis.intent})
            return await self._handle_data_query(account_number, query_analysis, user_message)
        elif query_analysis.intent == "transfer_money":
            logger.info({"action": "execute_money_transfer"})
            return await self._handle_money_transfer(account_number, query_analysis, user_message)
        else:
            logger.info({"action": "execute_general_query"})
            return await self._handle_general_query(user_message, first_name)

    async def _analyze_query(self, user_message: str, account_number: str) -> QueryResult:
        """Use LLM to analyze query and generate MongoDB pipeline."""
        logger.info({
            "action": "analyze_query",
            "user_message": user_message,
            "account_number": account_number
        })

        # Try improved fallback intent detection first
        try:
            intent, pipeline = self.detect_intent_fallback(user_message)
            if intent != "general":
                # Replace placeholder account_number in pipeline
                pipeline = self.replace_account_number_in_pipeline(pipeline, account_number)
                logger.info({
                    "action": "fallback_query_result",
                    "intent": intent,
                    "pipeline": pipeline
                })
                return QueryResult(intent=intent, pipeline=pipeline)
        except Exception as e:
            logger.error({
                "action": "fallback_intent_detection",
                "error": str(e),
                "user_message": user_message
            })

        # If fallback fails, use original LLM approach
        try:
            start_time = datetime.now()
            response = llm.invoke([
                SystemMessage(content=query_prompt.format(
                    user_message=user_message,
                    current_date=datetime.now().strftime("%Y-%m-%d")
                ))
            ])
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info({
                "action": "llm_response",
                "response_time_seconds": response_time,
                "response_content": response.content
            })

            result = self.extract_json_from_response(response.content)
            if result is None:
                logger.error({
                    "action": "llm_response_parse",
                    "error": "Could not extract JSON from response",
                    "raw_response": response.content
                })
                return QueryResult(intent="general", pipeline=[])

            if not isinstance(result, dict) or "intent" not in result:
                logger.error({
                    "action": "llm_response_validation",
                    "error": "Invalid LLM response structure",
                    "result": result
                })
                return QueryResult(intent="general", pipeline=[])

            # Replace placeholder account_number in pipeline
            pipeline = self.replace_account_number_in_pipeline(result.get("pipeline", []), account_number)

            query_result = QueryResult(
                intent=result.get("intent", "general"),
                pipeline=pipeline,
                response_format=result.get("response_format", "natural_language")
            )
            logger.info({
                "action": "query_result",
                "query_result": query_result.dict()
            })
            return query_result
        except Exception as e:
            logger.error({
                "action": "analyze_query",
                "error": str(e),
                "user_message": user_message
            })
            return QueryResult(intent="general", pipeline=[])

    async def _handle_balance_inquiry(self, account_number: str, first_name: str, query_analysis: QueryResult, user_message: str) -> str:
        """Handle balance inquiry."""
        logger.info({
            "action": "handle_balance_inquiry",
            "account_number": account_number,
            "first_name": first_name
        })
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/user_balance",
                    json={"account_number": account_number}
                )
                response.raise_for_status()
                data = response.json()
                logger.info({
                    "action": "balance_api_response",
                    "status_code": response.status_code,
                    "data": data
                })

                # Format response using LLM
                start_time = datetime.now()
                formatted_response = llm.invoke([
                    SystemMessage(content=response_prompt.format(
                        user_message=user_message,
                        intent="balance_inquiry",
                        data=json.dumps(data)
                    ))
                ])
                response_time = (datetime.now() - start_time).total_seconds()
                logger.info({
                    "action": "format_balance_response",
                    "response_time_seconds": response_time,
                    "formatted_response": formatted_response.content
                })
                return formatted_response.content
        except Exception as e:
            logger.error({
                "action": "handle_balance_inquiry",
                "error": str(e)
            })
            return "Error fetching balance. Please try again."

    async def _handle_data_query(self, account_number: str, query_analysis: QueryResult, user_message: str) -> str:
        """Handle queries requiring MongoDB pipeline execution."""
        logger.info({
            "action": "handle_data_query",
            "account_number": account_number,
            "intent": query_analysis.intent,
            "pipeline": query_analysis.pipeline
        })
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/execute_pipeline",
                    json={"account_number": account_number, "pipeline": query_analysis.pipeline}
                )
                response.raise_for_status()
                data = response.json()
                logger.info({
                    "action": "data_query_api_response",
                    "status_code": response.status_code,
                    "data": data
                })

                # Format response using LLM
                start_time = datetime.now()
                formatted_response = llm.invoke([
                    SystemMessage(content=response_prompt.format(
                        user_message=user_message,
                        intent=query_analysis.intent,
                        data=json.dumps(data)
                    ))
                ])
                response_time = (datetime.now() - start_time).total_seconds()
                logger.info({
                    "action": "format_data_query_response",
                    "response_time_seconds": response_time,
                    "formatted_response": formatted_response.content
                })
                return formatted_response.content
        except Exception as e:
            logger.error({
                "action": "handle_data_query",
                "error": str(e)
            })
            return "Error processing query. Please try again."

    async def _handle_money_transfer(self, account_number: str, query_analysis: QueryResult, user_message: str) -> str:
        """Handle money transfer requests."""
        logger.info({
            "action": "handle_money_transfer",
            "account_number": account_number,
            "user_message": user_message
        })
        try:
            start_time = datetime.now()
            response = llm.invoke([SystemMessage(content=transfer_prompt.format(user_message=user_message))])
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info({
                "action": "transfer_details_extraction",
                "response_time_seconds": response_time,
                "response_content": response.content
            })

            transfer_details = self.extract_json_from_response(response.content)
            if transfer_details is None:
                logger.error({
                    "action": "transfer_details_parse",
                    "error": "Could not extract JSON from response",
                    "raw_response": response.content
                })
                return "Invalid transfer details. Please try again."

            if not all(transfer_details.get(k) for k in ["amount", "currency", "recipient"]):
                logger.warning({
                    "action": "transfer_details_validation",
                    "status": "incomplete",
                    "transfer_details": transfer_details
                })
                return "Please specify the amount, currency (USD or PKR), and recipient."

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/transfer_money",
                    json={
                        "from_account": account_number,
                        "to_recipient": transfer_details["recipient"],
                        "amount": transfer_details["amount"],
                        "currency": transfer_details["currency"]
                    }
                )
                response.raise_for_status()
                data = response.json()
                logger.info({
                    "action": "transfer_api_response",
                    "status_code": response.status_code,
                    "data": data
                })

                # Format response using LLM
                start_time = datetime.now()
                formatted_response = llm.invoke([
                    SystemMessage(content=response_prompt.format(
                        user_message=f"Transfer {transfer_details['amount']} {transfer_details['currency']} to {transfer_details['recipient']}",
                        intent="transfer_money",
                        data=json.dumps(data)
                    ))
                ])
                response_time = (datetime.now() - start_time).total_seconds()
                logger.info({
                    "action": "format_transfer_response",
                    "response_time_seconds": response_time,
                    "formatted_response": formatted_response.content
                })
                return formatted_response.content
        except Exception as e:
            logger.error({
                "action": "handle_money_transfer",
                "error": str(e)
            })
            return "Error processing transfer. Please try again."

    async def _handle_general_query(self, user_message: str, first_name: str) -> str:
        """Handle general or unrecognized queries."""
        help_message = f"""
        Sorry, I couldn't understand your query. Could you please clarify? 
        I can assist with the following types of questions:
        - Check your balance: "What is my balance?"
        - View transactions: "Show my last 15 transactions"
        - Analyze spending: "How much did I spend on Netflix in June?"
        - category spending: "Where did I spend the most last month?"
        - Transfer money: "Transfer 500 USD to John"
        Please specify your request for assistance.
        """
        logger.info({
            "action": "handle_general_query",
            "user_message": user_message,
            "first_name": first_name,
            "response": help_message
        })
        return help_message