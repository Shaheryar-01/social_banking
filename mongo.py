from pymongo import MongoClient

# --- 1️⃣ Connect to MongoDB ---
client = MongoClient("mongodb://localhost:27017/")

# --- 2️⃣ Access database and collections ---
db = client["bank_database"]
users_col = db["users"]
bank_statements = db["bank_statements"]

print(db.list_collection_names())  # Optional: Print collection names to verify connection
