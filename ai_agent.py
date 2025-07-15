import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import jsonschema
import re

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
    model="gpt-4",
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
    vendor: Optional[str] = None
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

class BankingAIAgent:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        
        # Prompt for extracting filters from user query
        self.filter_extraction_prompt = PromptTemplate(
            input_variables=["user_message", "current_date"],
            template="""
            You are a banking AI assistant. Extract relevant filters from the user's query for MongoDB aggregation.
            
            Current date: {current_date}
            
            Available database fields:
            - account_number (string)
            - date (ISODate)
            - type (string: "debit" or "credit")
            - description (string: vendor/merchant name)
            - Category (string: Food, Entertainment, Travel, Finance, Shopping, etc.)
            - amount_usd (number)
            - amount_pkr (number)
            - balance_usd (number)
            - balance_pkr (number)
            
            Extract the following filters from the user query and return as JSON:
            {{
                "vendor": "vendor name if mentioned (e.g., Netflix, Uber, Amazon)",
                "category": "category if mentioned (e.g., Food, Entertainment, Travel)",
                "month": "month name if mentioned (e.g., january, june, december)",
                "year": "year if mentioned (default to 2025 if not specified)",
                "transaction_type": "debit or credit if specified",
                "amount_range": {{"min": number, "max": number}} if amount range mentioned,
                "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} if specific date range,
                "limit": number if specific count mentioned (e.g., last 10 transactions)
            }}
            
            Rules:
            - Only include fields that are explicitly mentioned or can be inferred
            - For vendor names, extract the exact name mentioned (case-insensitive matching will be handled later)
            - For months, use lowercase full names (january, february, etc.)
            - For spending queries, default transaction_type to "debit"
            - If "last X transactions" mentioned, set limit to X
            - If no year specified but month is mentioned, assume 2025
            - Return null for fields not mentioned
            
            Examples:
            
            Query: "how much did i spend on netflix in june"
            Response: {{
                "vendor": "netflix",
                "category": null,
                "month": "june",
                "year": 2025,
                "transaction_type": "debit",
                "amount_range": null,
                "date_range": null,
                "limit": null
            }}
            
            Query: "show my last 10 transactions"
            Response: {{
                "vendor": null,
                "category": null,
                "month": null,
                "year": null,
                "transaction_type": null,
                "amount_range": null,
                "date_range": null,
                "limit": 10
            }}
            
            Query: "how much did i spend on food last month"
            Response: {{
                "vendor": null,
                "category": "Food",
                "month": "june",
                "year": 2025,
                "transaction_type": "debit",
                "amount_range": null,
                "date_range": null,
                "limit": null
            }}
            
            Query: "transactions over 1000 USD in may"
            Response: {{
                "vendor": null,
                "category": null,
                "month": "may",
                "year": 2025,
                "transaction_type": null,
                "amount_range": {{"min": 1000}},
                "date_range": null,
                "limit": null
            }}
            
            User query: {user_message}
            Return only the JSON object.
            """
        )
        
        # Prompt for generating MongoDB pipeline from extracted filters
        self.pipeline_generation_prompt = PromptTemplate(
            input_variables=["filters", "intent", "account_number"],
            template="""
            Generate a MongoDB aggregation pipeline based on the extracted filters and intent.
            
            Account Number: {account_number}
            Intent: {intent}
            Extracted Filters: {filters}
            
            Generate a pipeline array with the following stages as needed:
            1. $match - for filtering documents
            2. $group - for aggregating data (spending analysis, category totals)
            3. $sort - for ordering results
            4. $limit - for limiting results
            5. $project - for selecting specific fields
            
            Rules:
            - Always include account_number in $match
            - For vendor matching, use $regex with case-insensitive option
            - For date filtering, convert month/year to ISODate range
            - For spending analysis, group by null and sum amounts
            - For transaction history, sort by date descending and _id descending
            - Use proper MongoDB syntax with ISODate objects
            
            Month to date range mapping (assuming year 2025):
            - january: 2025-01-01 to 2025-01-31
            - february: 2025-02-01 to 2025-02-28
            - march: 2025-03-01 to 2025-03-31
            - april: 2025-04-01 to 2025-04-30
            - may: 2025-05-01 to 2025-05-31
            - june: 2025-06-01 to 2025-06-30
            - july: 2025-07-01 to 2025-07-31
            - august: 2025-08-01 to 2025-08-31
            - september: 2025-09-01 to 2025-09-30
            - october: 2025-10-01 to 2025-10-31
            - november: 2025-11-01 to 2025-11-30
            - december: 2025-12-01 to 2025-12-31
            
            Examples:
            
            Intent: spending_analysis, Filters: {{"vendor": "netflix", "month": "june", "year": 2025, "transaction_type": "debit"}}
            Pipeline: [
                {{"$match": {{"account_number": "{account_number}", "type": "debit", "description": {{"$regex": "netflix", "$options": "i"}}, "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59Z"}}}}}}}},
                {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
            ]
            
            Intent: transaction_history, Filters: {{"limit": 10}}
            Pipeline: [
                {{"$match": {{"account_number": "{account_number}"}}}},
                {{"$sort": {{"date": -1, "_id": -1}}}},
                {{"$limit": 10}}
            ]
            
            Intent: category_spending, Filters: {{"category": "Food", "month": "june", "year": 2025, "transaction_type": "debit"}}
            Pipeline: [
                {{"$match": {{"account_number": "{account_number}", "type": "debit", "Category": "Food", "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59Z"}}}}}}}},
                {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
            ]
            
            Return only the JSON array pipeline.
            """
        )

        # Original query prompt (kept as fallback)
        self.query_prompt = PromptTemplate(
            input_variables=["user_message", "current_date"],
            template="""
            You are a banking AI assistant. Analyze the user's query and return a valid JSON response with:
            1. "intent" - one of: balance_inquiry, transaction_history, spending_analysis, category_spending, transfer_money, general
            2. "pipeline" - MongoDB aggregation pipeline to fetch the required data
            3. "response_format" - "natural_language"

            Current date: {current_date}

            MongoDB collections:
            - users: { "_id": ObjectId, "user_id": string, "first_name": string, "last_name": string, "dob": string, "mother_name": string, "place_of_birth": string, "account_number": string, "current_balance_usd": number, "current_balance_pkr": number }
            - bank_statements: { "_id": ObjectId, "account_number": string, "date": ISODate, "type": string ("debit"/"credit"), "description": string, "Category": string (e.g., Food, Entertainment), "amount_usd": number, "amount_pkr": number, "balance_usd": number, "balance_pkr": number }

            Guidelines:
            - For balance_inquiry, query the users collection or latest bank_statements document. Set pipeline to [].
            - For transaction_history, use $match, $sort, and optional $limit in the pipeline.
            - For spending_analysis, use $match and $group to aggregate spending by category, vendor, or date range.
            - For category_spending, use $match and $group for category aggregation.
            - For transfer_money, set pipeline to [] and handle via API.
            - Use ISODate for date filters (e.g., {{"$gte": ISODate("2025-06-01T00:00:00Z")}}).
            - For relative dates (e.g., "last month"), calculate appropriate ISODate ranges based on {current_date}.
            - Ensure the pipeline is valid MongoDB syntax and safe to execute.

            User query: {user_message}
            Return a valid JSON object.
            """
        )

        # Response formatting prompt
        self.response_prompt = PromptTemplate(
            input_variables=["user_message", "data", "intent"],
            template="""
            You are a banking AI assistant. Format the API response data into a natural language answer to the user's query. Be concise, professional, and avoid emojis or informal language.

            User query: {user_message}
            Intent: {intent}
            API response data: {data}

            Guidelines:
            - For balance_inquiry, report current balances in USD and PKR.
            - For transaction_history, list transactions with date, description, category, and non-zero amount (USD or PKR).
            - For spending_analysis, summarize total spending in USD and PKR, specifying the vendor or category if applicable.
            - For category_spending, list categories with amounts and percentages.
            - For transfer_money, confirm the transfer details or report errors.
            - For general, provide a helpful response explaining available queries.
            - If the data indicates an error (e.g., {{"status": "fail"}}), return a user-friendly error message.
            - For spending_analysis, if total_usd or total_pkr is zero, omit that currency from the response unless both are zero.

            Format the response for the query and data provided.
            """
        )

    def extract_filters_with_llm(self, user_message: str) -> FilterExtraction:
        """Use LLM to extract filters from user query."""
        try:
            logger.info({
                "action": "extract_filters_with_llm",
                "user_message": user_message
            })
            
            response = llm.invoke([
                SystemMessage(content=self.filter_extraction_prompt.format(
                    user_message=user_message,
                    current_date=datetime.now().strftime("%Y-%m-%d")
                ))
            ])
            
            logger.info({
                "action": "llm_filter_extraction_response",
                "response_content": response.content
            })
            
            try:
                filters_dict = json.loads(response.content)
                filters = FilterExtraction(**filters_dict)
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
                SystemMessage(content=self.pipeline_generation_prompt.format(
                    filters=json.dumps(filters.dict()),
                    intent=intent,
                    account_number=account_number
                ))
            ])
            
            logger.info({
                "action": "llm_pipeline_generation_response",
                "response_content": response.content
            })
            
            try:
                pipeline = json.loads(response.content)
                logger.info({
                    "action": "pipeline_generated",
                    "pipeline": pipeline
                })
                return pipeline
            except json.JSONDecodeError as e:
                logger.error({
                    "action": "pipeline_generation_parse_error",
                    "error": str(e),
                    "raw_response": response.content
                })
                return []
                
        except Exception as e:
            logger.error({
                "action": "generate_pipeline_from_filters",
                "error": str(e)
            })
            return []

    def detect_intent_from_filters(self, user_message: str, filters: FilterExtraction) -> str:
        """Detect intent based on user message and extracted filters."""
        user_message_lower = user_message.lower()
        
        if "balance" in user_message_lower:
            return "balance_inquiry"
        elif "transaction" in user_message_lower or filters.limit:
            return "transaction_history"
        elif "spend" in user_message_lower or "spent" in user_message_lower:
            if filters.category:
                return "category_spending"
            else:
                return "spending_analysis"
        elif "transfer" in user_message_lower:
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
                SystemMessage(content=self.query_prompt.format(
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

            try:
                result = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error({
                    "action": "llm_response_parse",
                    "error": str(e),
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
                    SystemMessage(content=self.response_prompt.format(
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
                    SystemMessage(content=self.response_prompt.format(
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
            # Extract transfer details using LLM
            transfer_prompt = PromptTemplate(
                input_variables=["user_message"],
                template="""
                Extract transfer details from the query:
                - amount: number
                - currency: "USD" or "PKR"
                - recipient: string
                Return JSON: {{"amount": number, "currency": string, "recipient": string}}
                
                Query: {user_message}
                """
            )
            start_time = datetime.now()
            response = llm.invoke([SystemMessage(content=transfer_prompt.format(user_message=user_message))])
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info({
                "action": "transfer_details_extraction",
                "response_time_seconds": response_time,
                "response_content": response.content
            })

            try:
                transfer_details = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error({
                    "action": "transfer_details_parse",
                    "error": str(e),
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
                    SystemMessage(content=self.response_prompt.format(
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
        Hello {first_name}, I can assist with the following:
        - Check your balance: "What is my balance?"
        - View transactions: "Show my last 15 transactions"
        - Analyze spending: "How much did I spend on Netflix in June?"
        - Category spending: "Where did I spend the most last month?"
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