from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mongo import users_col, bank_statements
from typing import Dict, Any, List, Optional
from datetime import datetime
from bson import ObjectId
import json
import re
import logging
from ai_agent import BankingAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize AI agent instance
ai_agent = BankingAIAgent()

class UserBalanceQuery(BaseModel):
    account_number: str

class MoneyTransferRequest(BaseModel):
    from_account: str
    to_recipient: str
    amount: float
    currency: str = "USD"

class PipelineQuery(BaseModel):
    account_number: str
    pipeline: List[Dict[str, Any]]

class VerifyRequest(BaseModel):
    account_number: str
    dob: str
    mother_name: str
    place_of_birth: str

class ProcessQueryRequest(BaseModel):
    user_message: str
    account_number: str
    first_name: str

class ProcessQueryResponse(BaseModel):
    status: str
    response: str
    error: Optional[str] = None

def convert_objectid_to_string(doc):
    """Recursively convert ObjectId to string in documents."""
    if isinstance(doc, dict):
        return {k: convert_objectid_to_string(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [convert_objectid_to_string(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc

def process_pipeline_dates(pipeline):
    """Process pipeline to handle date objects properly."""
    processed_pipeline = []
    for stage in pipeline:
        processed_stage = {}
        for key, value in stage.items():
            if isinstance(value, dict):
                processed_stage[key] = process_dict_dates(value)
            else:
                processed_stage[key] = value
        processed_pipeline.append(processed_stage)
    return processed_pipeline

def process_dict_dates(obj):
    """Recursively process dictionary to handle date objects."""
    if isinstance(obj, dict):
        processed = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                if "$date" in v:
                    try:
                        processed[k] = datetime.fromisoformat(v["$date"].replace("Z", "+00:00"))
                    except ValueError as e:
                        logger.error({
                            "action": "process_dict_dates",
                            "error": f"Invalid date format: {v['$date']}"
                        })
                        processed[k] = v  # Keep original value to avoid breaking pipeline
                elif "$gte" in v or "$lte" in v or "$lt" in v or "$gt" in v:
                    processed[k] = {}
                    for op, date_val in v.items():
                        if isinstance(date_val, dict) and "$date" in date_val:
                            try:
                                processed[k][op] = datetime.fromisoformat(date_val["$date"].replace("Z", "+00:00"))
                            except ValueError as e:
                                logger.error({
                                    "action": "process_dict_dates",
                                    "error": f"Invalid date format: {date_val['$date']}"
                                })
                                processed[k][op] = date_val
                        else:
                            processed[k][op] = date_val
                else:
                    processed[k] = process_dict_dates(v)
            elif isinstance(v, list):
                processed[k] = [process_dict_dates(item) for item in v]
            else:
                processed[k] = v
        return processed
    else:
        return obj


@router.post("/verify")
def verify_user(data: VerifyRequest):
    """Verify user credentials"""
    try:
        query = {
            "account_number": data.account_number.strip(),
            "dob": data.dob.strip(),
            "mother_name": {"$regex": f"^{re.escape(data.mother_name.strip())}$", "$options": "i"},
            "place_of_birth": {"$regex": f"^{re.escape(data.place_of_birth.strip())}$", "$options": "i"}
        }

        logger.info(f"🔍 Querying MongoDB with: {query}")
        user = users_col.find_one(query)

        if user:
            return {
                "status": "success",
                "user": {
                    "first_name": user["first_name"],
                    "account_number": user["account_number"]
                }
            }
        else:
            return {"status": "fail", "reason": "User not found"}
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return {"status": "fail", "reason": "Verification failed"}

@router.post("/user_balance")
async def get_user_balance(data: UserBalanceQuery):
    """Get user's current balance."""
    try:
        # First check if user exists
        user = users_col.find_one({"account_number": data.account_number})
        if not user:
            return {"status": "fail", "reason": "User not found"}
        
        # Get latest transaction for most current balance
        latest_txn = bank_statements.find_one(
            {"account_number": data.account_number},
            sort=[("date", -1), ("_id", -1)]  # Added _id for consistent sorting
        )
        
        if latest_txn:
            current_balance_usd = latest_txn.get("balance_usd", 0)
            current_balance_pkr = latest_txn.get("balance_pkr", 0)
        else:
            # Fallback to user document balance
            current_balance_usd = user.get("current_balance_usd", 0)
            current_balance_pkr = user.get("current_balance_pkr", 0)
        
        return {
            "status": "success",
            "user": {
                "first_name": user["first_name"],
                "last_name": user.get("last_name", ""),
                "account_number": user["account_number"],
                "current_balance_usd": current_balance_usd,
                "current_balance_pkr": current_balance_pkr
            }
        }
    except Exception as e:
        logger.error(f"Balance error: {e}")
        return {"status": "fail", "error": str(e)}

@router.post("/execute_pipeline")
async def execute_pipeline(data: PipelineQuery):
    """Execute a dynamic MongoDB aggregation pipeline."""
    try:
        # Validate input
        if not data.pipeline:
            return {"status": "fail", "reason": "Empty pipeline provided"}
        
        if not data.account_number:
            return {"status": "fail", "reason": "Account number is required"}
        
        # Process pipeline to handle date objects
        processed_pipeline = process_pipeline_dates(data.pipeline)
        
        logger.info(f"Executing pipeline for account {data.account_number}: {processed_pipeline}")
        
        # Execute pipeline on transactions collection (bank_statements)
        result = list(bank_statements.aggregate(processed_pipeline))
        
        # Convert ObjectId to string for JSON serialization
        result = convert_objectid_to_string(result)
        
        logger.info(f"Pipeline execution successful. Returned {len(result)} documents")
        
        return {
            "status": "success",
            "data": result,
            "count": len(result)
        }
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        return {"status": "fail", "error": str(e)}

@router.post("/transfer_money")
async def transfer_money(data: MoneyTransferRequest):
    """Handle money transfer."""
    try:
        # Validate input
        if data.amount <= 0:
            return {"status": "fail", "reason": "Transfer amount must be positive"}
        
        if data.currency not in ["USD", "PKR"]:
            return {"status": "fail", "reason": "Currency must be USD or PKR"}
        
        # Check sender exists
        sender = users_col.find_one({"account_number": data.from_account})
        if not sender:
            return {"status": "fail", "reason": "Sender account not found"}
        
        # Get current balance from latest transaction
        latest_txn = bank_statements.find_one(
            {"account_number": data.from_account},
            sort=[("date", -1), ("_id", -1)]
        )
        
        if latest_txn:
            current_balance_usd = latest_txn.get("balance_usd", 0)
            current_balance_pkr = latest_txn.get("balance_pkr", 0)
        else:
            current_balance_usd = sender.get("current_balance_usd", 0)
            current_balance_pkr = sender.get("current_balance_pkr", 0)
        
        # Check sufficient balance
        if data.currency == "USD" and current_balance_usd < data.amount:
            return {
                "status": "fail", 
                "reason": f"Insufficient USD balance. Available: ${current_balance_usd:.2f}, Required: ${data.amount:.2f}"
            }
        elif data.currency == "PKR" and current_balance_pkr < data.amount:
            return {
                "status": "fail", 
                "reason": f"Insufficient PKR balance. Available: ₨{current_balance_pkr:.2f}, Required: ₨{data.amount:.2f}"
            }
        
        # Calculate new balances
        new_balance_usd = current_balance_usd - (data.amount if data.currency == "USD" else 0)
        new_balance_pkr = current_balance_pkr - (data.amount if data.currency == "PKR" else 0)
        
        # Create transfer transaction
        transfer_txn = {
            "account_number": data.from_account,
            "date": datetime.now(),
            "type": "debit",
            "description": f"Transfer to {data.to_recipient}",
            "category": "Transfer",
            "amount_usd": data.amount if data.currency == "USD" else 0,
            "amount_pkr": data.amount if data.currency == "PKR" else 0,
            "balance_usd": new_balance_usd,
            "balance_pkr": new_balance_pkr
        }
        
        # Insert transaction
        txn_result = bank_statements.insert_one(transfer_txn)
        
        # Update user's current balance in users collection
        balance_update = {
            "current_balance_usd": new_balance_usd,
            "current_balance_pkr": new_balance_pkr
        }
        
        users_col.update_one(
            {"account_number": data.from_account},
            {"$set": balance_update}
        )
        
        logger.info(f"Transfer successful: {data.amount} {data.currency} from {data.from_account} to {data.to_recipient}")
        logger.info(f"Updated user balance - USD: {new_balance_usd}, PKR: {new_balance_pkr}")
        
        return {
            "status": "success",
            "message": f"Successfully transferred {data.amount} {data.currency} to {data.to_recipient}",
            "transaction_id": str(txn_result.inserted_id),
            "new_balance_usd": new_balance_usd,
            "new_balance_pkr": new_balance_pkr,
            "transfer_details": {
                "amount": data.amount,
                "currency": data.currency,
                "recipient": data.to_recipient,
                "timestamp": transfer_txn["date"].isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Transfer error: {e}")
        return {"status": "fail", "error": str(e)}

@router.post("/process_query", response_model=ProcessQueryResponse)
async def process_query(data: ProcessQueryRequest):
    """Process user banking queries using AI agent with contextual awareness."""
    try:
        logger.info({
            "action": "api_process_query_start",
            "user_message": data.user_message,
            "account_number": data.account_number,
            "first_name": data.first_name
        })
        
        # Use the AI agent to process the query
        response = await ai_agent.process_query(
            user_message=data.user_message,
            account_number=data.account_number,
            first_name=data.first_name
        )
        
        logger.info({
            "action": "api_process_query_success",
            "account_number": data.account_number,
            "response_length": len(response)
        })
        
        return ProcessQueryResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        logger.error({
            "action": "api_process_query_error",
            "error": str(e),
            "account_number": data.account_number,
            "user_message": data.user_message
        })
        
        return ProcessQueryResponse(
            status="error",
            response="Sorry, an error occurred while processing your request. Please try again.",
            error=str(e)
        )

# Additional utility endpoint for debugging
@router.post("/debug_pipeline")
async def debug_pipeline(data: PipelineQuery):
    """Debug endpoint to test pipeline processing."""
    try:
        processed_pipeline = process_pipeline_dates(data.pipeline)
        return {
            "status": "success",
            "original_pipeline": data.pipeline,
            "processed_pipeline": processed_pipeline
        }
    except Exception as e:
        logger.error(f"Debug pipeline error: {e}")
        return {"status": "fail", "error": str(e)}

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "banking_ai_backend"
    }