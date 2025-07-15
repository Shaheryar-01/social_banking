#webhook.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import os
import requests
from typing import Dict, Any
from state import authenticated_users, processed_messages
from ai_agent import BankingAIAgent
import time

app = FastAPI()

VERIFY_TOKEN = "helloworld3"
PAGE_ACCESS_TOKEN = "EAAbUiG1U0wYBPBHf5hXMclgmLXIs2O8pKbqt6Gc3uOW43NxC1ElQAKexFvBjseAfVZB1MGBLhsguN0IR155ZBwFx3fVDMzeDhSTzKjVJoTBuWSirs6m5FRQWbAR9foNMtcz2VUEagRCvZCazRtyZA6nGjZBMIySiUdO7xHWdU7ZA30nJXKI87bx5MWiZAG4AQKkVPFirDBlbAZDZD"

BACKEND_URL = "http://localhost:8000"

ai_agent = BankingAIAgent()

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

    return JSONResponse(content={"status": "ok"})

user_last_message_time = {}

async def process_user_message(sender_id: str, user_message: str) -> str:
    """Process user message using AI agent."""

    current_time = time.time()
    
    # Rate limiting - allow max 1 message per 2 seconds per user
    if sender_id in user_last_message_time:
        if current_time - user_last_message_time[sender_id] < 2:
            return "Please wait a moment before sending another message."
    
    user_last_message_time[sender_id] = current_time

    is_verified = sender_id in authenticated_users
    
    if not is_verified:
        if user_message.lower().startswith("verify"):
            return "Please provide your account number, date of birth, mother's name, and place of birth, separated by commas."
        
        elif "," in user_message and len(user_message.split(",")) == 4:
            return await handle_verification(sender_id, user_message)
        
        else:
            return "Please verify your identity by typing 'verify' to begin."

    user_data = authenticated_users[sender_id]
    account_number = user_data["account_number"]
    first_name = user_data["first_name"]
    
    try:
        response = await ai_agent.process_query(
            user_message=user_message,
            account_number=account_number,
            first_name=first_name
        )
        return response
    except Exception as e:
        print(f"Error processing query: {e}")
        return "Sorry, an error occurred while processing your request. Please try again."

async def handle_verification(sender_id: str, user_message: str) -> str:
    """Handle user verification."""
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
            return f"Hello {result['user']['first_name']}, you are now verified. You can ask about your balance, transactions, spending, or make transfers. For example, try: 'What is my current balance?' or 'How much did I spend on groceries last month?'"
        else:
            return "Verification failed. Please check your information and try again."
    
    except Exception as e:
        print(f"Verification error: {e}")
        return "Invalid format. Please provide: account_number, date of birth, mother's name, place of birth (comma-separated)."

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
        print("Sent:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error sending message: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)