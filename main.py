from fastapi import FastAPI
import verify, transactions

app = FastAPI()

app.include_router(verify.router)
app.include_router(transactions.router)
