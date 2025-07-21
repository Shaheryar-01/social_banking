import pandas as pd
from pymongo import MongoClient

# --- 1️⃣ Load Excel files using Pandas ---
# 🔁 Convert account_number to string
users_df = pd.read_excel("users_updated.xlsx")
# Convert account_number to string to ensure consistency
users_df["account_number"] = users_df["account_number"].astype(str)

transactions_df = pd.read_excel("final_transactions_database.xlsx")
# Convert account_number to string to ensure consistency
transactions_df["account_number"] = transactions_df["account_number"].astype(str)


# --- 2️⃣ Connect to MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["bank_database"]

# --- 3️⃣ Define collections ---
users_col = db["users"]
transactions_col = db["bank_statements"]

# --- 4️⃣ Clean insert (remove existing data) ---
users_col.delete_many({})
transactions_col.delete_many({})

# --- 5️⃣ Insert data into MongoDB ---
users_col.insert_many(users_df.to_dict(orient="records"))
transactions_col.insert_many(transactions_df.to_dict(orient="records"))

print("✅ Users and transactions successfully loaded to MongoDB.")
