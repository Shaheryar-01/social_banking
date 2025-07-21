# Updated webhook.py - API-based architecture with Professional Sage
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import os
import requests
from typing import Dict, Any
from state import authenticated_users, processed_messages, periodic_cleanup
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VERIFY_TOKEN = "helloworld3"
PAGE_ACCESS_TOKEN = "EAAOqZBb1DZCWYBPGiZCdRaVk6KrTAiQYclW4ZCZC9e8FiC4EqdOU0zN2gLDaVC1UtXeDXYT7VtnKPyr5NV3TZAgChtsMiDhzgZBsqk6eHZA8IKUQjqlORPXIatiTbs9OekNOeFxL16xOpEM2gJKMgJLR7yo70dPCHWBTyILXZAiBLEzQt9KfZBdOYCIEGyOVDdzMDM9aey"

BACKEND_URL = "http://localhost:8000"

# Professional Response Formatter for Sage
class ProfessionalResponseFormatter:
    """Handles professional, warm response formatting for Sage banking assistant."""
    
    def __init__(self):
        self.assistant_name = "Sage"
        
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

# Initialize formatter
response_formatter = ProfessionalResponseFormatter()

def get_professional_welcome_message(first_name: str) -> str:
    """Professional welcome message for newly verified users."""
    time_greeting = response_formatter.get_time_of_day_greeting()
    
    return f"""{time_greeting}, {first_name}! Welcome to your personal banking assistant. 🏦✨

I'm Sage, and I'm delighted to help you manage your banking needs today. You're now verified and ready to go!

I can help you with:

💰 **Account Management**
• Check your balance and account status  
• Review your spending patterns and insights

📊 **Financial Analysis**
• Analyze your expenses by category
• Track your spending trends over time
• Get personalized financial insights

📝 **Transaction Services**
• View your transaction history
• Search and filter your transactions
• Transfer money securely

I'm designed to understand context, so you can have natural conversations with me. For example, you can ask "How much did I spend last month?" and then follow up with "from this, how much on groceries?"

What would you like to explore first? I'm here to make your banking experience as smooth and insightful as possible!"""

@app.get("/webhook")
async def webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    else:
        raise HTTPException(status_code=403, detail="Invalid verification token.")


@app.post("/webhook")
async def receive_message(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)
    
    # Check if it's a valid webhook format
    if "entry" not in data:
        return JSONResponse(content={"status": "ok"})

    for entry in data.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            # Get message ID to prevent duplicates
            message_id = messaging_event.get("message", {}).get("mid")
            sender_id = messaging_event["sender"]["id"]

            # Skip if we've already processed this message
            if message_id and message_id in processed_messages:
                continue

            if "message" in messaging_event:
                # Add message ID to processed set
                if message_id:
                    processed_messages.add(message_id)
                
                user_message = messaging_event["message"].get("text", "")
                
                # Only process if there's actual text content
                if user_message.strip():
                    response_text = await process_user_message(sender_id, user_message)
                    send_message(sender_id, response_text)

    # Periodic cleanup every 100 messages
    if len(processed_messages) % 100 == 0:
        periodic_cleanup()

    return JSONResponse(content={"status": "ok"})

user_last_message_time = {}

async def process_user_message(sender_id: str, user_message: str) -> str:
    """Process user message using API calls to the backend with professional tone."""

    current_time = time.time()
    
    # Rate limiting - allow max 1 message per 2 seconds per user
    if sender_id in user_last_message_time:
        if current_time - user_last_message_time[sender_id] < 2:
            return "I appreciate your enthusiasm! Please give me just a moment to process your previous message before sending another. 😊"
    
    user_last_message_time[sender_id] = current_time

    is_verified = sender_id in authenticated_users
    
    if not is_verified:
        if user_message.lower().startswith("verify"):
            return """Hello! I'm Sage, your personal banking assistant. 👋

To get started, I'll need to verify your identity for security. Please provide the following information separated by commas:

📝 **Required Information:**
• Account number
• Date of birth (YYYY-MM-DD format)  
• Mother's maiden name
• Place of birth

**Example format:**
1001, 1990-01-15, Jane Smith, New York

Once verified, I'll be able to help you with all your banking needs!"""
        
        elif "," in user_message and len(user_message.split(",")) == 4:
            return await handle_verification(sender_id, user_message)
        
        else:
            return """Hello! Welcome to your personal banking assistant. 🏦

For your security, I need to verify your identity first. Please type 'verify' to begin the verification process.

I'm here to help you manage your accounts, analyze your spending, and handle transfers once you're verified!"""

    user_data = authenticated_users[sender_id]
    account_number = user_data["account_number"]
    first_name = user_data["first_name"]
    
    try:
        # Log the query for debugging
        logger.info({
            "action": "processing_user_query",
            "sender_id": sender_id,
            "account_number": account_number,
            "user_message": user_message
        })
        
        # Make API call to backend process_query endpoint
        response = await call_process_query_api(
            user_message=user_message,
            account_number=account_number,
            first_name=first_name
        )
        
        logger.info({
            "action": "query_processed_successfully",
            "sender_id": sender_id,
            "response_length": len(response)
        })
        
        return response
    except Exception as e:
        logger.error({
            "action": "process_user_message_error",
            "sender_id": sender_id,
            "error": str(e),
            "user_message": user_message
        })
        return f"I apologize, {first_name}, but I encountered a technical issue while processing your request. Please try again, and I'll be happy to help you!"

async def call_process_query_api(user_message: str, account_number: str, first_name: str) -> str:
    """Make API call to backend process_query endpoint."""
    try:
        payload = {
            "user_message": user_message,
            "account_number": account_number,
            "first_name": first_name
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:  # 60 second timeout for AI processing
            response = await client.post(
                f"{BACKEND_URL}/process_query",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result["status"] == "success":
                return result["response"]
            else:
                logger.error({
                    "action": "process_query_api_error",
                    "error": result.get("error", "Unknown error"),
                    "account_number": account_number
                })
                return result.get("response", "Sorry, I couldn't process your request. Please try again.")
                
    except httpx.TimeoutException:
        logger.error({
            "action": "process_query_api_timeout",
            "account_number": account_number,
            "user_message": user_message
        })
        return "Request timed out. Please try again with a simpler query."
        
    except httpx.HTTPStatusError as e:
        logger.error({
            "action": "process_query_api_http_error",
            "status_code": e.response.status_code,
            "account_number": account_number,
            "error": str(e)
        })
        return "Backend service error. Please try again later."
        
    except Exception as e:
        logger.error({
            "action": "process_query_api_unexpected_error",
            "error": str(e),
            "account_number": account_number
        })
        return "Unexpected error occurred. Please try again."

async def handle_verification(sender_id: str, user_message: str) -> str:
    """Handle user verification with professional response."""
    try:
        acc, dob, mom, pob = [x.strip() for x in user_message.split(",")]
        payload = {
            "account_number": acc,
            "dob": dob,
            "mother_name": mom,
            "place_of_birth": pob
        }

        async with httpx.AsyncClient() as client:
            res = await client.post(f"{BACKEND_URL}/verify", json=payload)
            result = res.json()

        if result["status"] == "success":
            authenticated_users[sender_id] = {
                "account_number": acc,
                "first_name": result["user"]["first_name"]
            }
            
            logger.info({
                "action": "user_verified_successfully",
                "sender_id": sender_id,
                "account_number": acc,
                "first_name": result["user"]["first_name"]
            })
            
            # Use professional welcome message
            return get_professional_welcome_message(result["user"]["first_name"])
        else:
            logger.warning({
                "action": "verification_failed",
                "sender_id": sender_id,
                "reason": result.get("reason", "Unknown")
            })
            return """I apologize, but I wasn't able to verify your account with the information provided. 

Please double-check your details and try again with the format:
account_number, date_of_birth, mother_name, place_of_birth

For example: 1001, 1990-01-15, Jane Smith, New York

If you continue to have trouble, please contact our support team for assistance."""
    
    except Exception as e:
        logger.error({
            "action": "verification_error",
            "sender_id": sender_id,
            "error": str(e)
        })
        return """I encountered an issue with the verification format. Please provide your information in this order, separated by commas:

account_number, date_of_birth, mother_name, place_of_birth

For example: 1001, 1990-01-15, Jane Smith, New York

I'm here to help once you're verified!"""

def send_message(recipient_id, message_text):
    """Send response to Facebook Messenger."""
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info({
            "action": "message_sent_successfully",
            "recipient_id": recipient_id,
            "response_status": response.status_code
        })
    except requests.exceptions.RequestException as e:
        logger.error({
            "action": "send_message_error",
            "recipient_id": recipient_id,
            "error": str(e)
        })

# Health check endpoint for webhook service
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test connection to backend
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            backend_healthy = response.status_code == 200
    except:
        backend_healthy = False
    
    return {
        "status": "healthy",
        "backend_connection": "healthy" if backend_healthy else "unhealthy",
        "timestamp": time.time(),
        "service": "banking_webhook"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)