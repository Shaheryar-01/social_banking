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

class ConversationContext(BaseModel):
    """Store conversation context for each user."""
    last_query: Optional[str] = None
    last_intent: Optional[str] = None
    last_filters: Optional[FilterExtraction] = None
    last_pipeline: Optional[List[Dict[str, Any]]] = None
    last_result: Optional[Dict[str, Any]] = None
    last_response: Optional[str] = None
    timestamp: Optional[datetime] = None

class ContextualQuery(BaseModel):
    """Result of contextual query analysis."""
    needs_context: bool = False
    has_reference: bool = False
    is_complete: bool = True
    missing_info: List[str] = Field(default_factory=list)
    clarification_needed: Optional[str] = None
    resolved_query: Optional[str] = None

class ProfessionalResponseFormatter:
    """Handles professional, warm response formatting for Sage banking assistant."""
    
    def __init__(self):
        self.assistant_name = "Sage"  # Your banking assistant's name
        
    def get_time_of_day_greeting(self):
        """Get appropriate greeting based on time of day."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 21:
            return "Good evening"
        else:
            return "Hello"
    
    def get_professional_response_prompt(self, user_message: str, intent: str, data: str, first_name: str, 
                                       is_contextual: bool = False) -> str:
        """Generate enhanced professional response prompt."""
        
        time_greeting = self.get_time_of_day_greeting()
        
        # Different prompts for different intents
        if intent == "balance_inquiry":
            return self._get_balance_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
        elif intent in ["spending_analysis", "category_spending"]:
            return self._get_spending_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
        elif intent == "transaction_history":
            return self._get_transaction_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
        elif intent == "transfer_money":
            return self._get_transfer_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
        else:
            return self._get_general_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
    
    def _get_balance_response_prompt(self, user_message: str, data: str, first_name: str, 
                                   time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a warm and professional personal banking assistant. You're helping {first_name}, a valued client.

        User Query: "{user_message}"
        Balance Data: {data}
        
        Response Style:
        - Warm and professional tone
        - Use {first_name}'s name appropriately (not excessively)
        - Start with "{time_greeting}, {first_name}!" if this is a fresh conversation
        - If contextual follow-up, use transitions like "Absolutely!" or "Of course!"
        
        Instructions:
        1. Present the balance information clearly and positively
        2. Add reassuring context about account standing
        3. Offer helpful next steps or additional services
        4. Use formatting for better readability (bullets, sections)
        5. End with a warm offer to help further
        
        Example tone: "I'm pleased to share that your current account balance is excellent. Your account is in great standing, and I'm here if you need any other assistance today."
        
        Make it feel like talking to a trusted personal banker who genuinely cares about helping.
        """
    
    def _get_spending_response_prompt(self, user_message: str, data: str, first_name: str, 
                                    time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a warm and professional personal banking assistant helping {first_name}.

        User Query: "{user_message}"
        Spending Data: {data}
        Is Contextual Follow-up: {is_contextual}
        
        Response Style:
        - Warm, helpful, and insightful
        - If first interaction: "{time_greeting}, {first_name}!"
        - If contextual: "Perfect!" or "Great question!" or "Let me break that down for you!"
        
        Instructions:
        1. Present spending information with helpful context and insights
        2. Add percentage calculations where relevant (e.g., "This represents X% of your total spending")
        3. Provide spending velocity context (daily/weekly averages)
        4. Compare to typical patterns when possible
        5. Offer proactive follow-up suggestions:
           - "Would you like me to break this down by category?"
           - "I can show you how this compares to last month"
           - "Would you like to see which merchants you spent the most at?"
        6. Use positive, encouraging language even for higher spending
        7. Format clearly with bullets or sections for complex information
        
        Tone Examples:
        - "I've analyzed your spending for that period..."
        - "Looking at your expenses during..."
        - "This gives you a good overview of..."
        - "To put this in perspective..."
        
        Always end with offering additional help or insights.
        """
    
    def _get_transaction_response_prompt(self, user_message: str, data: str, first_name: str, 
                                       time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a professional banking assistant helping {first_name} review their transactions.

        User Query: "{user_message}"
        Transaction Data: {data}
        Is Contextual Follow-up: {is_contextual}
        
        Response Style:
        - Professional but friendly
        - If first interaction: "{time_greeting}, {first_name}!"
        - If contextual: "Here's what I found!" or "Let me pull that up for you!"
        
        Instructions:
        1. Present transactions in a clear, organized format
        2. Highlight key insights (largest transaction, most frequent merchant, etc.)
        3. Group related transactions when helpful
        4. Point out any interesting patterns or unusual activity (tactfully)
        5. Offer additional analysis:
           - "Would you like me to filter these by category?"
           - "I can show you just the larger transactions if that's helpful"
           - "Would you like to see how these compare to your typical spending?"
        6. Use clear formatting (dates, amounts, descriptions)
        
        Make the transaction review feel like a helpful financial advisor reviewing activity with care and attention.
        """
    
    def _get_transfer_response_prompt(self, user_message: str, data: str, first_name: str, 
                                    time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a trusted banking assistant helping {first_name} with their money transfer.

        User Query: "{user_message}"
        Transfer Result: {data}
        
        Response Style:
        - Professional, reassuring, and detail-oriented
        - Celebratory tone for successful transfers
        - Clear and organized information presentation
        
        Instructions:
        1. Confirm transfer success with enthusiasm ("Excellent!" or "Perfect!")
        2. Present transfer details in organized format:
           - ✅ Transfer Completed
           - Amount and currency
           - Recipient
           - Transaction ID
           - New balance
        3. Provide helpful timing information (when funds will be available)
        4. Reassure about security and completion
        5. Offer additional assistance
        
        Example structure:
        "Excellent! I've successfully processed your transfer. Here are the details for your records:
        
        ✅ **Transfer Completed**
        • Amount: $X
        • Recipient: [Name]
        • Transaction ID: [ID]
        
        Your new balance is [amount]. The transfer should reflect in the recipient's account within 2-3 business hours."
        
        End with offering further assistance.
        """
    
    def _get_general_response_prompt(self, user_message: str, data: str, first_name: str, 
                                   time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a helpful banking assistant. Respond to {first_name}'s query in a warm, professional manner.

        User Query: "{user_message}"
        Available Information: {data}
        
        Provide a helpful, professional response that offers guidance and additional assistance.
        Use {time_greeting}, {first_name} as appropriate.
        """

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
        # Store conversation contexts for each user
        self.user_contexts: Dict[str, ConversationContext] = {}
        self.response_formatter = ProfessionalResponseFormatter()  # Add this line
        
    def get_user_context(self, account_number: str) -> ConversationContext:
        """Get or create conversation context for a user."""
        if account_number not in self.user_contexts:
            self.user_contexts[account_number] = ConversationContext()
        return self.user_contexts[account_number]
    
    def update_user_context(self, account_number: str, query: str, intent: str, 
                           filters: FilterExtraction, pipeline: List[Dict[str, Any]], 
                           result: Dict[str, Any] = None, response: str = None):
        """Update user's conversation context."""
        context = self.get_user_context(account_number)
        context.last_query = query
        context.last_intent = intent
        context.last_filters = filters
        context.last_pipeline = pipeline
        context.last_result = result
        context.last_response = response
        context.timestamp = datetime.now()
        
        logger.info({
            "action": "update_user_context",
            "account_number": account_number,
            "context_updated": True
        })
    
    def analyze_contextual_query(self, user_message: str, account_number: str) -> ContextualQuery:
        """Analyze if query needs context and what information is missing using LLM."""
        context = self.get_user_context(account_number)
    
        # First, use LLM to detect if this is a contextual query
        contextual_detection_result = self._detect_contextual_reference_with_llm(user_message, context)
    
        # If no contextual reference detected, analyze as standalone query
        if not contextual_detection_result["is_contextual"]:
            return self._analyze_standalone_query(user_message)
    
        # If contextual reference detected but no previous context exists
        if not context.last_query or not context.last_filters:
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=False,
                clarification_needed="I don't have any previous query to reference. Could you please provide the complete information for your request?"
            )
    
        # Check if context is recent (within last 10 minutes)
        if context.timestamp and (datetime.now() - context.timestamp).seconds > 600:
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=False,
                clarification_needed="The previous context is too old. Could you please provide the complete information for your request?"
            )
    
        # Try to resolve the query with context
        try:
            resolved_query = self._resolve_contextual_query_with_llm(user_message, context)
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=True,
                resolved_query=resolved_query
            )
        except Exception as e:
            logger.error(f"Error resolving contextual query: {e}")
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=False,
                clarification_needed="I couldn't understand the context. Could you please provide the complete information for your request?"
            )
  
    def _detect_contextual_reference_with_llm(self, user_message: str, context: ConversationContext) -> Dict[str, Any]:
        """Use LLM to detect if user query references previous context."""
    
        # Prepare context summary for LLM
        context_summary = "No previous context available"
        if context.last_query and context.last_response:
            context_summary = f"""
            Previous Query: "{context.last_query}"
            Previous Response Summary: "{context.last_response[:200]}..."
            Previous Intent: "{context.last_intent}"
            """
    
        contextual_detection_prompt = f"""
        Analyze if the current user query is referencing or building upon previous conversation context.

        Current Query: "{user_message}"
    
        Previous Context:
        {context_summary}

        A query is contextual if it:
        1. References previous results (e.g., "from this", "from that data", "those transactions")
        2. Uses pronouns that refer to previous content (e.g., "them", "these", "it")
        3. Asks for filtering/drilling down into previous results (e.g., "show me the grocery ones", "which were over $100")
        4. Uses relative terms that depend on previous context (e.g., "the highest", "the recent ones")
        5. Asks follow-up questions that only make sense with previous context

        Examples of CONTEXTUAL queries:
        - "from this how much on groceries" (after showing spending data)
        - "show me the grocery ones" (after showing transactions)
        - "which of these are over $100" (after showing a list)
        - "the highest transaction" (after showing transactions)
        - "break it down by category" (after showing total spending)
        - "filter them by last week" (after showing results)

        Examples of NON-CONTEXTUAL queries:
        - "show me my balance" (complete standalone query)
        - "how much did I spend on groceries in June" (complete with all details)
        - "transfer 500 USD to John" (complete transfer request)
        - "show me transactions from last week" (complete with timeframe)

        Return JSON:
        {{
            "is_contextual": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation of why this is or isn't contextual"
        }}
        """
    
        try:
            response = llm.invoke([SystemMessage(content=contextual_detection_prompt)])
            result = self.extract_json_from_response(response.content)
        
            if result and isinstance(result, dict):
                logger.info({
                    "action": "contextual_detection_llm",
                    "user_message": user_message,
                    "result": result
                })
                return result
            else:
                # Fallback to safe assumption
                return {
                    "is_contextual": False,
                    "confidence": 0.5,
                    "reasoning": "Could not parse LLM response"
                }
            
        except Exception as e:
            logger.error(f"Error in contextual detection with LLM: {e}")
            # Fallback to trigger word detection
            return self._fallback_trigger_word_detection(user_message)



    def _fallback_trigger_word_detection(self, user_message: str) -> Dict[str, Any]:
        """Fallback method using trigger words if LLM fails."""
        context_phrases = [
            "from this", "from that", "out of this", "out of that", "from the above",
            "from these", "from those", "of this", "of that", "in this", "in that",
            "them", "these", "those", "it", "they", "break it down", "filter them",
            "show me the", "which ones", "the highest", "the lowest", "the recent ones"
        ]
    
        has_reference = any(phrase in user_message.lower() for phrase in context_phrases)
    
        return {
            "is_contextual": has_reference,
            "confidence": 0.7 if has_reference else 0.8,
            "reasoning": f"Trigger word detection: {'found' if has_reference else 'not found'} contextual phrases"
        }

    def _resolve_contextual_query_with_llm(self, user_message: str, context: ConversationContext) -> str:
        """Enhanced contextual query resolution using LLM."""
    
        resolution_prompt = f"""
        You are helping resolve a contextual banking query. The user is referencing previous conversation context.
    
        Current User Query: "{user_message}"
    
        Previous Context:
        - Previous Query: "{context.last_query}"
        - Previous Intent: "{context.last_intent}"
        - Previous Filters Applied: {json.dumps(context.last_filters.dict() if context.last_filters else {})}
        - Previous Response Summary: "{(context.last_response or '')[:300]}..."
    
        Your task is to combine the current query with the previous context to create a complete, standalone query.
    
        Guidelines:
        1. Preserve all relevant filters from the previous context
        2. Add new filtering/analysis requested in current query
        3. Make the query completely self-contained
        4. Maintain the original intent unless explicitly changed
    
        Examples:
    
        Previous: "how much did I spend in June"
        Current: "from this how much on groceries"
        Resolved: "how much did I spend on groceries in June"
    
        Previous: "show me my transactions from last week"
        Current: "which ones are over $100"
        Resolved: "show me my transactions from last week that are over $100"
    
        Previous: "my spending analysis for May 2024"
        Current: "break it down by category"
        Resolved: "show me my spending breakdown by category for May 2024"
    
        Previous: "show me restaurant transactions in March"
        Current: "the highest one"
        Resolved: "show me the highest restaurant transaction in March"
    
        Return ONLY the resolved query as a plain string, no JSON or formatting.
        """
    
        try:
            response = llm.invoke([SystemMessage(content=resolution_prompt)])
            resolved_query = response.content.strip()
        
            # Remove any quotes or extra formatting
            resolved_query = resolved_query.strip('"\'')
        
            logger.info({
                "action": "resolve_contextual_query_llm",
                "original_query": user_message,
                "resolved_query": resolved_query,
                "context_query": context.last_query
            })
        
            return resolved_query
        
        except Exception as e:
            logger.error(f"Error resolving contextual query with LLM: {e}")
            raise e






    def _analyze_standalone_query(self, user_message: str) -> ContextualQuery:
        """Simplified analysis - only check transfers for completeness."""
    
        # Use LLM only for transfer queries
        if any(word in user_message.lower() for word in ["transfer", "send", "pay", "wire", "remit"]):
        
            completeness_prompt = f"""
            Analyze this transfer query for completeness:
        
            Query: "{user_message}"
        
            Check if the query has:
            1. Amount (e.g., $500, 50000 PKR, etc.)
            2. Recipient (e.g., "to John", "to account 1234", etc.)
        
            Examples:
            ❌ "transfer money" → INCOMPLETE (missing amount and recipient)
            ❌ "transfer $50" → INCOMPLETE (missing recipient)  
            ❌ "send money to John" → INCOMPLETE (missing amount)
            ✅ "transfer $50 to John" → COMPLETE
            ✅ "send 500 USD to account 1234" → COMPLETE
        
            Return JSON:
            {{
                "is_complete": true/false,
                "missing_info": ["amount", "recipient"],
                "clarification_needed": "What specific information to ask for"
            }}
            """
        
            try:
                response = llm.invoke([SystemMessage(content=completeness_prompt)])
                result = self.extract_json_from_response(response.content)
            
                if result:
                    return ContextualQuery(
                        needs_context=False,
                        has_reference=False,
                        is_complete=result.get("is_complete", True),
                        missing_info=result.get("missing_info", []),
                        clarification_needed=result.get("clarification_needed")
                    )
            except Exception as e:
                logger.error(f"Error analyzing transfer completeness: {e}")
    
        # For all non-transfer queries, always return complete
        return ContextualQuery(is_complete=True)

    
    def _simple_completeness_check(self, user_message: str) -> ContextualQuery:
        """Simplified completeness check - only ask for clarification on transfers."""
        
    
        user_message_lower = user_message.lower()
    
        # Check for transfer queries - ONLY type that needs clarification
        transfer_words = ["transfer", "send", "pay", "wire", "remit"]
        has_transfer = any(word in user_message_lower for word in transfer_words)
    
        if has_transfer:
            # Check what's missing in transfer query
            missing_info = []
            clarifications = []
        
            # Check for amount
            amount_pattern = r'\$?\d+(?:\.\d{2})?(?:\s*(?:usd|pkr|dollars?|rupees?))?'
            if not re.search(amount_pattern, user_message_lower):
                missing_info.append("amount")
                clarifications.append("the amount (e.g., '$500' or '50000 PKR')")
        
            # Check for recipient
            recipient_indicators = ["to ", "for ", "recipient", "account"]
            has_recipient = any(indicator in user_message_lower for indicator in recipient_indicators)
            if not has_recipient:
                missing_info.append("recipient")
                clarifications.append("the recipient (e.g., 'to John' or 'to account 1234')")
        
            if missing_info:
                clarification_text = f"Please specify {' and '.join(clarifications)}."
                return ContextualQuery(
                    is_complete=False,
                    missing_info=missing_info,
                    clarification_needed=clarification_text
                )
    
        # For ALL other queries (spending, transactions, balance, etc.)
        # Always return complete - no clarification needed
        return ContextualQuery(is_complete=True)

    
    def _resolve_contextual_query(self, user_message: str, context: ConversationContext) -> str:
        """Resolve a contextual query using previous context."""
        
        resolution_prompt = f"""
        Resolve this contextual banking query using the previous context:
        
        Current Query: "{user_message}"
        
        Previous Context:
        - Last Query: "{context.last_query}"
        - Last Intent: "{context.last_intent}"
        - Last Filters: {json.dumps(context.last_filters.dict() if context.last_filters else {})}
        - Last Response: "{context.last_response or 'No response available'}"
        
        The user is referencing the previous query/result. Combine the context with the new query to create a complete, standalone query.
        
        Examples:
        - Previous: "how much did I spend in June" → Current: "from this how much on groceries" 
          → Resolved: "how much did I spend on groceries in June"
        
        - Previous: "show me transactions in May" → Current: "from these which ones are over $100"
          → Resolved: "show me transactions in May that are over $100"
        
        Return only the resolved query as a string, no JSON or extra formatting.
        """
        
        try:
            response = llm.invoke([SystemMessage(content=resolution_prompt)])
            resolved_query = response.content.strip()
            
            logger.info({
                "action": "resolve_contextual_query",
                "original_query": user_message,
                "resolved_query": resolved_query,
                "context_query": context.last_query
            })
            
            return resolved_query
            
        except Exception as e:
            logger.error(f"Error resolving contextual query: {e}")
            raise e
    
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
        """Process user banking queries with contextual awareness."""
        logger.info({
            "action": "process_query",
            "user_message": user_message,
            "account_number": account_number,
            "first_name": first_name
        })

        # Step 1: Analyze if query needs context or clarification
        contextual_analysis = self.analyze_contextual_query(user_message, account_number)
        
        # If query is incomplete and needs clarification
        if not contextual_analysis.is_complete and contextual_analysis.clarification_needed:
            logger.info({
                "action": "requesting_clarification",
                "user_message": user_message,
                "missing_info": contextual_analysis.missing_info
            })
            return contextual_analysis.clarification_needed
        
        # Step 2: Use resolved query if available, otherwise use original
        query_to_process = contextual_analysis.resolved_query or user_message
        
        logger.info({
            "action": "processing_resolved_query",
            "original_query": user_message,
            "resolved_query": query_to_process
        })

        # Step 3: Analyze the query (resolved or original)
        query_analysis = await self._analyze_query(query_to_process, account_number)
        logger.info({
            "action": "query_analysis_result",
            "query_analysis": query_analysis.dict(),
            "resolved_query": query_to_process
        })

        # Step 4: Validate pipeline
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

        # Step 5: Execute appropriate action
        result = None
        response = None
        
        if query_analysis.intent == "balance_inquiry":
            logger.info({"action": "execute_balance_inquiry"})
            response = await self._handle_balance_inquiry(account_number, first_name, query_analysis, query_to_process)
        elif query_analysis.intent in ["transaction_history", "spending_analysis", "category_spending"]:
            logger.info({"action": "execute_data_query", "intent": query_analysis.intent})
            response = await self._handle_data_query(account_number, query_analysis, query_to_process, first_name)
        elif query_analysis.intent == "transfer_money":
            logger.info({"action": "execute_money_transfer"})
            response = await self._handle_money_transfer(account_number, query_analysis, query_to_process, first_name)
        else:
            logger.info({"action": "execute_general_query"})
            response = await self._handle_general_query(query_to_process, first_name)

        # Step 6: Update context for future queries
        if query_analysis.intent != "general":
            self.update_user_context(
                account_number=account_number,
                query=query_to_process,
                intent=query_analysis.intent,
                filters=query_analysis.filters or FilterExtraction(),
                pipeline=query_analysis.pipeline,
                result=result,
                response=response
            )

        return response

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
        """Enhanced balance inquiry with professional responses."""
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
                
                # Check if this is a contextual follow-up
                context = self.get_user_context(account_number)
                is_contextual = context.last_query is not None and context.timestamp and \
                               (datetime.now() - context.timestamp).seconds < 300  # 5 minutes
                
                # Use professional response formatter
                professional_prompt = self.response_formatter.get_professional_response_prompt(
                    user_message=user_message,
                    intent="balance_inquiry",
                    data=json.dumps(data),
                    first_name=first_name,
                    is_contextual=is_contextual
                )
                
                formatted_response = llm.invoke([SystemMessage(content=professional_prompt)])
                return formatted_response.content
                
        except Exception as e:
            logger.error({"action": "handle_balance_inquiry", "error": str(e)})
            return f"I apologize, {first_name}, but I'm experiencing a technical issue retrieving your balance. Please try again in a moment, and I'll be happy to help!"

    async def _handle_data_query(self, account_number: str, query_analysis: QueryResult, user_message: str, first_name: str) -> str:
        """Enhanced data query handling with professional responses."""
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
                
                # Check if contextual
                context = self.get_user_context(account_number)
                is_contextual = context.last_query is not None and context.timestamp and \
                               (datetime.now() - context.timestamp).seconds < 300
                
                # Use professional response formatter
                professional_prompt = self.response_formatter.get_professional_response_prompt(
                    user_message=user_message,
                    intent=query_analysis.intent,
                    data=json.dumps(data),
                    first_name=first_name,
                    is_contextual=is_contextual
                )
                
                formatted_response = llm.invoke([SystemMessage(content=professional_prompt)])
                return formatted_response.content
                
        except Exception as e:
            logger.error({"action": "handle_data_query", "error": str(e)})
            return "I apologize, but I encountered an issue processing your request. Please try again, and I'll be happy to help you with your query!"

    async def _handle_money_transfer(self, account_number: str, query_analysis: QueryResult, user_message: str, first_name: str) -> str:
        """Enhanced money transfer with professional responses."""
        logger.info({
            "action": "handle_money_transfer",
            "account_number": account_number,
            "user_message": user_message
        })
        try:
            start_time = datetime.now()
            response = llm.invoke([SystemMessage(content=transfer_prompt.format(user_message=user_message))])
            
            transfer_details = self.extract_json_from_response(response.content)
            if transfer_details is None:
                return f"I'm sorry, {first_name}, but I couldn't understand the transfer details. Could you please specify the amount, currency (USD or PKR), and recipient? For example: 'Transfer 500 USD to John Smith'."

            if not all(transfer_details.get(k) for k in ["amount", "currency", "recipient"]):
                return f"To complete your transfer, {first_name}, I need a bit more information. Please specify the amount, currency (USD or PKR), and recipient. For example: 'Transfer 500 USD to John Smith'."

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
                
                # Use professional response formatter
                professional_prompt = self.response_formatter.get_professional_response_prompt(
                    user_message=f"Transfer {transfer_details['amount']} {transfer_details['currency']} to {transfer_details['recipient']}",
                    intent="transfer_money",
                    data=json.dumps(data),
                    first_name=first_name,
                    is_contextual=False
                )
                
                formatted_response = llm.invoke([SystemMessage(content=professional_prompt)])
                return formatted_response.content
                
        except Exception as e:
            logger.error({"action": "handle_money_transfer", "error": str(e)})
            return f"I apologize, {first_name}, but I encountered an issue processing your transfer. Your account security is our priority, so please try again or contact support if the issue persists."

    async def _handle_general_query(self, user_message: str, first_name: str) -> str:
        """Enhanced general query handling with professional tone."""
        time_greeting = self.response_formatter.get_time_of_day_greeting()
        
        help_message = f"""
        {time_greeting}, {first_name}! I'd be happy to help you with your banking needs.
        
        I can assist you with:
        
        💰 **Account Information**
        • Check your current balance
        • Review recent account activity
        
        📊 **Spending Analysis** 
        • Analyze your spending patterns
        • Break down expenses by category
        • Compare spending across different time periods
        
        📝 **Transaction History**
        • View your recent transactions
        • Filter transactions by date or amount
        • Search for specific merchants or categories
        
        💸 **Money Transfers**
        • Transfer funds to other accounts
        • Send money to friends and family
        
        You can ask me questions like:
        • "What's my current balance?"
        • "How much did I spend on groceries last month?"
        • "Show me my transactions from this week"
        • "Transfer 500 USD to John Smith"
        
        I'm also great with follow-up questions! After I show you information, you can ask "from this, how much on utilities?" or similar contextual questions.
        
        What would you like to know about your account today?
        """
        
        logger.info({
            "action": "handle_general_query",
            "user_message": user_message,
            "first_name": first_name
        })
        return help_message