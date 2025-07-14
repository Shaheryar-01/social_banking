from fastapi import APIRouter
from pydantic import BaseModel
from mongo import transactions_col

router = APIRouter()

class TransactionQuery(BaseModel):
    account_number: str

@router.post("/transactions")
def get_transactions(data: TransactionQuery):
    txns = list(transactions_col.find({"account_number": data.account_number}).sort("date", -1).limit(5))
    for t in txns:
        t["_id"] = str(t["_id"])  # Convert ObjectId to string
    return {"transactions": txns}
