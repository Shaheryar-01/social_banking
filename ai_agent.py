import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
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
            - description (string: Zong, Grocery Store, Careem, Foodpanda, Amazon, JazzCash, Utility Bill, McDonalds, Salary, Daraz, Netflix, ABC Corp, Recieved from X (replace X with name))
            - category (string: Telecom, Groceries, Travel, Food, Shopping, Finance, Utilities, Income,Entertainment, Salary, Transfer)

            - amount_usd (number)
            - amount_pkr (number)
            - balance_usd (number)
            - balance_pkr (number)
            
            Extract the following filters from the user query and return as JSON:
            {{
                "description": "description name if mentioned (e.g., Netflix, Uber, Amazon)",
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
            - For description names, extract the exact name mentioned (case-insensitive matching will be handled later)
            - For months, use lowercase full names (january, february, etc.)
            - For spending queries, default transaction_type to "debit"
            - If "last X transactions" mentioned, set limit to X
            - If no year specified but month is mentioned, assume 2025
            - Return null for fields not mentioned
            - The account supports multiple currencies: `amount_usd` and `amount_pkr` represent independent transaction amounts in USD and PKR, respectively, and are not converted versions of each other.     Similarly, `balance_usd` and `balance_pkr` are separate balances maintained in each currency.
            
            Examples:
            
            Query: "how much did i spend on netflix in june"
            Response: {{
                "description": "netflix",
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
                "description": null,
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
                "description": null,
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
                "description": null,
                "category": null,
                "month": "may",
                "year": 2025,
                "transaction_type": null,
                "amount_range": {{"min": 1000}},
                "date_range": null,
                "limit": null
            }}
            
            User query: {user_message}
            ### RESPONSE FORMAT – READ CAREFULLY
            Return **exactly one** valid JSON value that fits the schema above.
            • No Markdown, no ``` fences, no comments, no keys other than the schema.
            • Do not pretty‑print; a single‑line minified object/array is required.
            • If a value is unknown, use null.
            Your entire reply must be parsable by `json.loads`.

            """
        )
        
        # Prompt for generating MongoDB pipeline from extracted filters
        # Enhanced prompt for generating MongoDB pipeline from extracted filters
        self.pipeline_generation_prompt = PromptTemplate(
        input_variables=["filters", "intent", "account_number"],
        template="""
        Generate a MongoDB aggregation pipeline based on the extracted filters and intent.

        IMPORTANT: Return ONLY the JSON array, no explanatory text, no markdown formatting.

        Account Number: {account_number}
        Intent: {intent}
        Extracted Filters: {filters}

        Generate a pipeline array with the following stages as needed:
        1. $match - for filtering documents
        2. $group - for aggregating data (spending analysis, category totals)
        3. $sort - for ordering results
        4. $limit - for limiting results
        5. $project - for selecting specific fields

        CRITICAL DATE HANDLING RULES:
        - If filters contain 'date_range' with start and end dates, use EXACT date range: {{"$gte": {{"$date": "START_DATET00:00:00Z"}}, "$lte": {{"$date": "END_DATET23:59:59Z"}}}}
        - If filters contain BOTH 'month' and 'year' (both not null), use full month range
        - If filters contain ONLY 'year' without month or date_range, DO NOT add any date filter
        - If filters contain null/empty month AND null/empty date_range, DO NOT add any date filter regardless of year
        - ALWAYS prioritize date_range over month/year when both are present
        - For single-day queries, start and end dates will be the same - use both start and end times

        General Rules:
        - Always include account_number in $match
        - For description and category matching, use $regex with case-insensitive option (e.g., {{"$regex": "value", "$options": "i"}})
        - For date filtering, convert to ISODate range in the format {{"$date": "YYYY-MM-DDTHH:mm:ssZ"}}
        - For spending analysis or category_spending, group by null and sum amounts
        - For transaction history, sort by date descending and _id descending
        - Ensure all ISODate values are valid and properly formatted (e.g., {{"$date": "2025-06-01T00:00:00Z"}})
        - Do not use incomplete or invalid syntax like "ISODate" or partial date strings
        - For category_spending intent with a category filter, always use {{"$regex": "<category>", "$options": "i"}} for the category field to ensure case-insensitive matching
        - Treat `amount_usd` and `amount_pkr` as independent fields representing transactions in their respective currencies

        DATE PROCESSING LOGIC:
        1. CHECK for date_range in filters first
        2. If date_range exists and is not null:
           - Use start date + "T00:00:00Z" for $gte
           - Use end date + "T23:59:59Z" for $lte
        3. If NO date_range but BOTH month AND year exist and are not null:
           - Use month to date range mapping below
        4. If month is null OR date_range is null:
           - DO NOT add any date filter to the pipeline
        5. Year alone (without month or date_range) should NOT create a date filter

        Month to date range mapping (for when date_range is NOT provided):
        - january: {{"$date": "2025-01-01T00:00:00Z"}} to {{"$date": "2025-01-31T23:59:59Z"}}
        - february: {{"$date": "2025-02-01T00:00:00Z"}} to {{"$date": "2025-02-28T23:59:59Z"}}
        - march: {{"$date": "2025-03-01T00:00:00Z"}} to {{"$date": "2025-03-31T23:59:59Z"}}
        - april: {{"$date": "2025-04-01T00:00:00Z"}} to {{"$date": "2025-04-30T23:59:59Z"}}
        - may: {{"$date": "2025-05-01T00:00:00Z"}} to {{"$date": "2025-05-31T23:59:59Z"}}
        - june: {{"$date": "2025-06-01T00:00:00Z"}} to {{"$date": "2025-06-30T23:59:59Z"}}
        - july: {{"$date": "2025-07-01T00:00:00Z"}} to {{"$date": "2025-07-31T23:59:59Z"}}
        - august: {{"$date": "2025-08-01T00:00:00Z"}} to {{"$date": "2025-08-31T23:59:59Z"}}
        - september: {{"$date": "2025-09-01T00:00:00Z"}} to {{"$date": "2025-09-30T23:59:59Z"}}
        - october: {{"$date": "2025-10-01T00:00:00Z"}} to {{"$date": "2025-10-31T23:59:59Z"}}
        - november: {{"$date": "2025-11-01T00:00:00Z"}} to {{"$date": "2025-11-30T23:59:59Z"}}
        - december: {{"$date": "2025-12-01T00:00:00Z"}} to {{"$date": "2025-12-31T23:59:59Z"}}

        Examples:

        Intent: category_spending, Filters: {{"category": "Shopping", "year": 2025, "transaction_type": "debit", "month": null, "date_range": null}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "category": {{"$regex": "Shopping", "$options": "i"}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: spending_analysis, Filters: {{"description": "netflix", "year": 2025, "transaction_type": "debit", "month": null, "date_range": null}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "description": {{"$regex": "netflix", "$options": "i"}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: spending_analysis, Filters: {{"description": "netflix", "month": "june", "year": 2025, "transaction_type": "debit"}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "description": {{"$regex": "netflix", "$options": "i"}}, "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59Z"}}}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: category_spending, Filters: {{"date_range": {{"start": "2025-06-01", "end": "2025-06-01"}}, "transaction_type": "debit"}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-01T23:59:59Z"}}}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: category_spending, Filters: {{"date_range": {{"start": "2025-06-01", "end": "2025-06-05"}}, "transaction_type": "debit"}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-05T23:59:59Z"}}}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: transaction_history, Filters: {{"limit": 10}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}"}}}},
            {{"$sort": {{"date": -1, "_id": -1}}}},
            {{"$limit": 10}}
        ]

        Intent: category_spending, Filters: {{"category": "Telecom", "month": "june", "year": 2025, "transaction_type": "debit"}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "category": {{"$regex": "Telecom", "$options": "i"}}, "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59Z"}}}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        STEP-BY-STEP PROCESSING:
        1. Check if 'date_range' exists in filters and is not null
        2. If yes, create date filter using start/end dates with proper time components
        3. If no, check if BOTH 'month' and 'year' exist and are not null
        4. If both month and year exist, use monthly range mapping
        5. If month is null or date_range is null, DO NOT add date filter
        6. Apply transaction_type filter if present
        7. Apply description/category regex filters if present
        8. Add appropriate aggregation stages based on intent

        Return only the JSON array pipeline.
        ### RESPONSE FORMAT – READ CAREFULLY
        Return **exactly one** valid JSON value that fits the schema above.
        • No Markdown, no ``` fences, no comments, no keys other than the schema.
        • Do not pretty‑print; a single‑line minified object/array is required.
        • If a value is unknown, use null.
        Your entire reply must be parsable by `json.loads`.

        """
    )
        # Response formatting prompt
        self.response_prompt = PromptTemplate(
            input_variables=["user_message", "data", "intent"],
            template="""
            consider yourself as  a consultant for a organization creating banking ai agents. Format the API response data into a natural language answer to the user's query. Be concise, professional, or informal language.

            User query: {user_message}
            Intent: {intent}
            API response data: {data}

            Guidelines:
            - For balance_inquiry, report current balances in USD and PKR.
            - For transaction_history, list transactions with date, description, category, and non-zero amount (USD or PKR).
            - For spending_analysis, summarize total spending in USD and PKR, specifying the description or category if applicable.
            - For category_spending, list categories with amounts and percentages.
            - For transfer_money, confirm the transfer details or report errors.
            - For general, provide a helpful response explaining available queries.
            - If the data indicates an error (e.g., {{"status": "fail"}}), return a user-friendly error message.
            - For spending_analysis, if total_usd or total_pkr is zero, omit that currency from the response unless both are zero.
            - When reporting amounts or balances, treat USD and PKR values as independent. Report both `amount_usd` and `amount_pkr` (or `balance_usd` and `balance_pkr`) when non-zero, and clarify that these are separate currency accounts, not conversions.

            Convert it into a finished and professional message.
            Format the response for the query and data provided.
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

            Example document in users document:
            {
            "_id": {
                "$oid": "6874e7bcdfb730a4127a09d1"
            },
            "user_id": "u005",
            "first_name": "Hamza",
            "last_name": "Sheikh",
            "dob": "1993-10-07",
            "mother_name": "Nuzhat",
            "place_of_birth": "Islamabad",
            "account_number": "1005",
            "current_balance_usd": 167952.5,
            "current_balance_pkr": 179325.41
            }

            - bank_statements: { "_id": ObjectId, "account_number": string, "date": ISODate, "type": string ("debit"/"credit"), "description": string, "category": string (e.g., Food, Entertainment), "amount_usd": number, "amount_pkr": number, "balance_usd": number, "balance_pkr": number }

            Example document in bank_statements collection:
            {
            "_id": {
                "$oid": "6874e7bcdfb730a4127a09d8"
            },
            "account_number": "1001",
            "date": {
                "$date": "2025-06-01T00:00:00.000Z"
            },
            "type": "debit",
            "description": "Grocery Store",
            "category": "Groceries",
            "amount_usd": 8468.68,
            "amount_pkr": 0,
            "balance_usd": 37747.04,
            "balance_pkr": 24519.44
            }

            Guidelines:
            - For current balance_inquiry, query the users collection or latest bank_statements document. Set pipeline to [].
            - For transaction_history, use $match, $sort, and optional $limit in the pipeline.
            - For spending_analysis, use $match and $group to aggregate spending by category, description, or date range.
            - For category_spending, use $match and $group for category aggregation.
            - For transfer_money, set pipeline to [] and handle via API.
            - Use ISODate for date filters (e.g., {{"$gte": ISODate("2025-06-01T00:00:00Z")}}).
            - For relative dates (e.g., "last month"), calculate appropriate ISODate ranges based on {current_date}.
            - Ensure the pipeline is valid MongoDB syntax and safe to execute.
            - The account maintains separate USD and PKR balances and transaction amounts. `amount_usd` and `amount_pkr` are independent, as are `balance_usd` and `balance_pkr`. Do not assume any conversion between these fields.

            User query: {user_message}
           ### RESPONSE FORMAT – READ CAREFULLY
            Return **exactly one** valid JSON value that fits the schema above.
            • No Markdown, no ``` fences, no comments, no keys other than the schema.
            • Do not pretty‑print; a single‑line minified object/array is required.
            • If a value is unknown, use null.
            Your entire reply must be parsable by `json.loads`.

            """
        )

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
                SystemMessage(content=self.pipeline_generation_prompt.format(
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
            # Create intent classification prompt
            intent_prompt = PromptTemplate(
                input_variables=["user_message", "filters"],
                template="""
                You are a banking AI assistant. Analyze the user's query and classify it into one of these intents:

                Available intents:
                1. "balance_inquiry" - User wants to check their account balance
                Examples: "What's my balance?", "How much money do I have?", "Check my account balance", "Show current balance"
                
                2. "transaction_history" - User wants to see their transaction history/list
                Examples: "Show my transactions", "List my recent purchases", "What are my last 10 transactions?", "Transaction history"
                
                3. "spending_analysis" - User wants to analyze their spending on specific items/merchants
                Examples: "How much did I spend on Netflix?", "What did I spend on Amazon last month?", "My Netflix expenses", "How much money went to Uber?"
                
                4. "category_spending" - User wants to analyze spending by category
                Examples: "How much did I spend on food?", "My entertainment expenses", "Food spending last month", "How much on groceries?"
                
                5. "transfer_money" - User wants to transfer money to someone
                Examples: "Transfer money to John", "Send $100 to Alice", "Pay my friend", "Transfer funds"
                
                6. "general" - General queries or unclear intent
                Examples: "Hello", "Help me", "What can you do?", unclear requests

                Classification rules:
                - If user mentions checking balance, money amount, or account status → "balance_inquiry"
                - If user asks for transaction list, history, or recent activities → "transaction_history"
                - If user asks about spending on specific merchants/services (Netflix, Amazon, etc.) → "spending_analysis"
                - If user asks about spending in categories (food, entertainment, etc.) → "category_spending"
                - If user wants to send/transfer money → "transfer_money"
                - If intent is unclear or general → "general"

                Consider these extracted filters to help classification:
                - If filters.limit is set → likely "transaction_history"
                - If filters.description is set → likely "spending_analysis"
                - If filters.category is set → likely "category_spending"
                - If filters.transaction_type is "debit" and specific merchant → likely "spending_analysis"

                User query: "{user_message}"
                Extracted filters: {filters}

                Respond with only the intent name (e.g., "balance_inquiry", "spending_analysis", etc.)
                """
            )
            
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
                - currency: usd/USD = "USD" or pkr/PKR = "PKR" (even if user types currency in lower case always extract it in upper case)
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