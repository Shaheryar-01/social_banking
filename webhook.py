from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import os
import requests
from state import authenticated_users  # Stores sessions as {sender_id: {user_data}}

app = FastAPI()

# ✅ Facebook credentials
VERIFY_TOKEN = "helloworld3"
PAGE_ACCESS_TOKEN = "EAAOqZBb1DZCWYBPJFK1Vpnhqpcv4z7gdijoUXXi1lqdyllBamRzdPVjPmxpONRdzTz18hXiRvJvpXarDX1ZAy71Lp6RXZC0rfZAOhFtY7cJGFxAw6Qabq3XBRjCcvCC8LRoYC7ad7zTjJdfZBY1fcD9ZAwZCLb2EmLBDa25ZAZBcZCWBZA7V0M6ZCewER6akH8QxYZAcAZBTR4B"

# ✅ Backend API base URL
BACKEND_URL = "http://localhost:8000"

# ✅ Facebook Webhook Verification
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

# ✅ Facebook Message Receiver
@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()

    for entry in data.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            sender_id = messaging_event["sender"]["id"]

            if "message" in messaging_event:
                user_message = messaging_event["message"].get("text", "").lower()

                # Step 1: Ask user to verify
                if user_message.startswith("verify"):
                    response_text = "🪪 Please send your account number, DOB, mother's name, and place of birth (comma-separated)."

                # Step 2: Handle verification
                elif "," in user_message and len(user_message.split(",")) == 4:
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
                        # ✅ Save verified user in session
                        authenticated_users[sender_id] = {
                            "account_number": acc,
                            "first_name": result["user"]["first_name"]
                        }

                        response_text = f"✅ Hi {result['user']['first_name']}, you're verified! You can now ask about your transactions."
                    else:
                        response_text = "❌ Verification failed. Please check your info."

                # Step 3: Transaction history
                elif "transaction" in user_message:
                    if sender_id not in authenticated_users:
                        response_text = "🔐 Please verify yourself first using 'verify'."
                    else:
                        account_number = authenticated_users[sender_id]["account_number"]
                        payload = {"account_number": account_number}

                        async with httpx.AsyncClient() as client:
                            res = await client.post(f"{BACKEND_URL}/transactions", json=payload)
                            txns = res.json().get("transactions", [])

                        if not txns:
                            response_text = "❌ No transactions found."
                        else:
                            message = "🧾 Last 5 transactions:\n"
                            for t in txns[:5]:  # Just show latest 5
                                date = str(t.get("date", "")).split("T")[0]
                                desc = t.get("description", "")
                                amt = t.get("amount", "")
                                currency = t.get("currency", "")
                                message += f"- {date} | {desc} | {amt} {currency}\n"
                            response_text = message

                # Step 4: Default fallback
                else:
                    response_text = "❓ I didn’t understand. Type 'verify' to get started."

                # ✅ Send the response
                send_message(sender_id, response_text)

    return JSONResponse(content={"status": "ok"})






# ✅ Function to send response to Facebook Messenger
def send_message(recipient_id, message_text):
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"

    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    print("✅ Sent:", response.json())
