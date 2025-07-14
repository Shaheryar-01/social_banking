from fastapi import APIRouter
from pydantic import BaseModel
from mongo import users_col
import re

router = APIRouter()

class VerifyRequest(BaseModel):
    account_number: str
    dob: str
    mother_name: str
    place_of_birth: str

@router.post("/verify")
def verify_user(data: VerifyRequest):
    query = {
        "account_number": data.account_number.strip(),
        "dob": data.dob.strip(),
        "mother_name": {"$regex": f"^{re.escape(data.mother_name.strip())}$", "$options": "i"},
        "place_of_birth": {"$regex": f"^{re.escape(data.place_of_birth.strip())}$", "$options": "i"}
    }

    print("🔍 Querying MongoDB with:", query)
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
