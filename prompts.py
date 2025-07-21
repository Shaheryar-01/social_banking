from langchain.prompts import PromptTemplate

filter_extraction_prompt = PromptTemplate(
            input_variables=["user_message", "current_date"],
            template="""
            You are a banking AI assistant. Extract relevant filters from the user's query for MongoDB aggregation.
            
            Current date: {current_date}
            
            Available database fields:
            - account_number (string)
            - date (ISODate)
            - type (string: "debit" or "credit")
            - description (string: Zong, Grocery Store, Careem, Foodpanda, Amazon, JazzCash, Utility Bill, McDonalds, Salary, Daraz, Netflix, ABC Corp, Recieved from X (replace X with name))
            - category (string: Telecom, Groceries, Travel, Food, Shopping, Finance, Utilities, Income,Entertainment, Salary, Transfer)

            - amount_usd (number)
            - amount_pkr (number)
            - balance_usd (number)
            - balance_pkr (number)
            
            Extract the following filters from the user query and return as JSON:
            {{
                "description": "description name if mentioned (e.g., Netflix, Uber, Amazon)",
                "category": "category if mentioned (e.g., Food, Entertainment, Travel)",
                "month": "month name if mentioned (e.g., january, june, december)",
                "year": "year if mentioned (default to 2025 if not specified)",
                "transaction_type": "debit or credit if specified",
                "amount_range": {{"min": number, "max": number}} if amount range mentioned,
                "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} if specific date range,
                "limit": number if specific count mentioned (e.g., last 10 transactions)
            }}
            
            Rules:
            - Only include fields that are explicitly mentioned or can be inferred
            - For description names, extract the exact name mentioned (case-insensitive matching will be handled later)
            - For months, use lowercase full names (january, february, etc.)
            - For spending queries, default transaction_type to "debit"
            - If "last X transactions" mentioned, set limit to X
            - If no year specified but month is mentioned, assume 2025
            - Return null for fields not mentioned
            - The account supports multiple currencies: `amount_usd` and `amount_pkr` represent independent transaction amounts in USD and PKR, respectively, and are not converted versions of each other.     Similarly, `balance_usd` and `balance_pkr` are separate balances maintained in each currency.
            
            Examples:
            
            Query: "how much did i spend on netflix in june"
            Response: {{
                "description": "netflix",
                "category": null,
                "month": "june",
                "year": 2025,
                "transaction_type": "debit",
                "amount_range": null,
                "date_range": null,
                "limit": null
            }}
            
            Query: "show my last 10 transactions"
            Response: {{
                "description": null,
                "category": null,
                "month": null,
                "year": null,
                "transaction_type": null,
                "amount_range": null,
                "date_range": null,
                "limit": 10
            }}
            
            Query: "how much did i spend on food last month"
            Response: {{
                "description": null,
                "category": "Food",
                "month": "june",
                "year": 2025,
                "transaction_type": "debit",
                "amount_range": null,
                "date_range": null,
                "limit": null
            }}
            
            Query: "transactions over 1000 USD in may"
            Response: {{
                "description": null,
                "category": null,
                "month": "may",
                "year": 2025,
                "transaction_type": null,
                "amount_range": {{"min": 1000}},
                "date_range": null,
                "limit": null
            }}
            
            User query: {user_message}
            ### RESPONSE FORMAT – READ CAREFULLY
            Return **exactly one** valid JSON value that fits the schema above.
            • No Markdown, no ``` fences, no comments, no keys other than the schema.
            • Do not pretty‑print; a single‑line minified object/array is required.
            • If a value is unknown, use null.
            Your entire reply must be parsable by `json.loads`.

            """
        )


pipeline_generation_prompt = PromptTemplate(
        input_variables=["filters", "intent", "account_number"],
        template="""
        Generate a MongoDB aggregation pipeline based on the extracted filters and intent.

        IMPORTANT: Return ONLY the JSON array, no explanatory text, no markdown formatting.

        Account Number: {account_number}
        Intent: {intent}
        Extracted Filters: {filters}

        Generate a pipeline array with the following stages as needed:
        1. $match - for filtering documents
        2. $group - for aggregating data (spending analysis, category totals)
        3. $sort - for ordering results
        4. $limit - for limiting results
        5. $project - for selecting specific fields

        CRITICAL DATE HANDLING RULES:
        - If filters contain 'date_range' with start and end dates, use EXACT date range: {{"$gte": {{"$date": "START_DATET00:00:00Z"}}, "$lte": {{"$date": "END_DATET23:59:59Z"}}}}
        - If filters contain BOTH 'month' and 'year' (both not null), use full month range
        - If filters contain ONLY 'year' without month or date_range, DO NOT add any date filter
        - If filters contain null/empty month AND null/empty date_range, DO NOT add any date filter regardless of year
        - ALWAYS prioritize date_range over month/year when both are present
        - For single-day queries, start and end dates will be the same - use both start and end times

        General Rules:
        - Always include account_number in $match
        - For description and category matching, use $regex with case-insensitive option (e.g., {{"$regex": "value", "$options": "i"}})
        - For date filtering, convert to ISODate range in the format {{"$date": "YYYY-MM-DDTHH:mm:ssZ"}}
        - For spending analysis or category_spending, group by null and sum amounts
        - For transaction history, sort by date descending and _id descending
        - Ensure all ISODate values are valid and properly formatted (e.g., {{"$date": "2025-06-01T00:00:00Z"}})
        - Do not use incomplete or invalid syntax like "ISODate" or partial date strings
        - For category_spending intent with a category filter, always use {{"$regex": "<category>", "$options": "i"}} for the category field to ensure case-insensitive matching
        - Treat `amount_usd` and `amount_pkr` as independent fields representing transactions in their respective currencies

        DATE PROCESSING LOGIC:
        1. CHECK for date_range in filters first
        2. If date_range exists and is not null:
           - Use start date + "T00:00:00Z" for $gte
           - Use end date + "T23:59:59Z" for $lte
        3. If NO date_range but BOTH month AND year exist and are not null:
           - Use month to date range mapping below
        4. If month is null OR date_range is null:
           - DO NOT add any date filter to the pipeline
        5. Year alone (without month or date_range) should NOT create a date filter

        Month to date range mapping (for when date_range is NOT provided):
        - january: {{"$date": "2025-01-01T00:00:00Z"}} to {{"$date": "2025-01-31T23:59:59Z"}}
        - february: {{"$date": "2025-02-01T00:00:00Z"}} to {{"$date": "2025-02-28T23:59:59Z"}}
        - march: {{"$date": "2025-03-01T00:00:00Z"}} to {{"$date": "2025-03-31T23:59:59Z"}}
        - april: {{"$date": "2025-04-01T00:00:00Z"}} to {{"$date": "2025-04-30T23:59:59Z"}}
        - may: {{"$date": "2025-05-01T00:00:00Z"}} to {{"$date": "2025-05-31T23:59:59Z"}}
        - june: {{"$date": "2025-06-01T00:00:00Z"}} to {{"$date": "2025-06-30T23:59:59Z"}}
        - july: {{"$date": "2025-07-01T00:00:00Z"}} to {{"$date": "2025-07-31T23:59:59Z"}}
        - august: {{"$date": "2025-08-01T00:00:00Z"}} to {{"$date": "2025-08-31T23:59:59Z"}}
        - september: {{"$date": "2025-09-01T00:00:00Z"}} to {{"$date": "2025-09-30T23:59:59Z"}}
        - october: {{"$date": "2025-10-01T00:00:00Z"}} to {{"$date": "2025-10-31T23:59:59Z"}}
        - november: {{"$date": "2025-11-01T00:00:00Z"}} to {{"$date": "2025-11-30T23:59:59Z"}}
        - december: {{"$date": "2025-12-01T00:00:00Z"}} to {{"$date": "2025-12-31T23:59:59Z"}}

        Examples:

        Intent: category_spending, Filters: {{"category": "Shopping", "year": 2025, "transaction_type": "debit", "month": null, "date_range": null}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "category": {{"$regex": "Shopping", "$options": "i"}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: spending_analysis, Filters: {{"description": "netflix", "year": 2025, "transaction_type": "debit", "month": null, "date_range": null}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "description": {{"$regex": "netflix", "$options": "i"}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: spending_analysis, Filters: {{"description": "netflix", "month": "june", "year": 2025, "transaction_type": "debit"}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "description": {{"$regex": "netflix", "$options": "i"}}, "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59Z"}}}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: category_spending, Filters: {{"date_range": {{"start": "2025-06-01", "end": "2025-06-01"}}, "transaction_type": "debit"}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-01T23:59:59Z"}}}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: category_spending, Filters: {{"date_range": {{"start": "2025-06-01", "end": "2025-06-05"}}, "transaction_type": "debit"}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-05T23:59:59Z"}}}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        Intent: transaction_history, Filters: {{"limit": 10}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}"}}}},
            {{"$sort": {{"date": -1, "_id": -1}}}},
            {{"$limit": 10}}
        ]

        Intent: category_spending, Filters: {{"category": "Telecom", "month": "june", "year": 2025, "transaction_type": "debit"}}
        Pipeline: [
            {{"$match": {{"account_number": "{account_number}", "type": "debit", "category": {{"$regex": "Telecom", "$options": "i"}}, "date": {{"$gte": {{"$date": "2025-06-01T00:00:00Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59Z"}}}}}}}},
            {{"$group": {{"_id": null, "total_usd": {{"$sum": "$amount_usd"}}, "total_pkr": {{"$sum": "$amount_pkr"}}}}}}
        ]

        STEP-BY-STEP PROCESSING:
        1. Check if 'date_range' exists in filters and is not null
        2. If yes, create date filter using start/end dates with proper time components
        3. If no, check if BOTH 'month' and 'year' exist and are not null
        4. If both month and year exist, use monthly range mapping
        5. If month is null or date_range is null, DO NOT add date filter
        6. Apply transaction_type filter if present
        7. Apply description/category regex filters if present
        8. Add appropriate aggregation stages based on intent

        Return only the JSON array pipeline.
        ### RESPONSE FORMAT – READ CAREFULLY
        Return **exactly one** valid JSON value that fits the schema above.
        • No Markdown, no ``` fences, no comments, no keys other than the schema.
        • Do not pretty‑print; a single‑line minified object/array is required.
        • If a value is unknown, use null.
        Your entire reply must be parsable by `json.loads`.

        """
    )
 

response_prompt = PromptTemplate(
            input_variables=["user_message", "data", "intent"],
            template="""
            consider yourself as  a consultant for a organization creating banking ai agents. Format the API response data into a natural language answer to the user's query. Be concise, professional, or informal language.

            User query: {user_message}
            Intent: {intent}
            API response data: {data}

            Guidelines:
            - For balance_inquiry, report current balances in USD and PKR.
            - For transaction_history, list transactions with date, description, category, and non-zero amount (USD or PKR).
            - For spending_analysis, summarize total spending in USD and PKR, specifying the description or category if applicable.
            - For category_spending, list categories with amounts and percentages.
            - For transfer_money, confirm the transfer details or report errors.
            - For general, provide a helpful response explaining available queries.
            - If the data indicates an error (e.g., {{"status": "fail"}}), return a user-friendly error message.
            - For spending_analysis, if total_usd or total_pkr is zero, omit that currency from the response unless both are zero.
            - When reporting amounts or balances, treat USD and PKR values as independent. Report both `amount_usd` and `amount_pkr` (or `balance_usd` and `balance_pkr`) when non-zero, and clarify that these are separate currency accounts, not conversions.

            Convert it into a finished and professional message.
            Format the response for the query and data provided.
            """
        )

query_prompt = PromptTemplate(
            input_variables=["user_message", "current_date"],
            template="""
            You are a banking AI assistant. Analyze the user's query and return a valid JSON response with:
            1. "intent" - one of: balance_inquiry, transaction_history, spending_analysis, category_spending, transfer_money, general
            2. "pipeline" - MongoDB aggregation pipeline to fetch the required data
            3. "response_format" - "natural_language"

            Current date: {current_date}

            MongoDB collections:
            - users: { "_id": ObjectId, "user_id": string, "first_name": string, "last_name": string, "dob": string, "mother_name": string, "place_of_birth": string, "account_number": string, "current_balance_usd": number, "current_balance_pkr": number }

            Example document in users document:
            {
            "_id": {
                "$oid": "6874e7bcdfb730a4127a09d1"
            },
            "user_id": "u005",
            "first_name": "Hamza",
            "last_name": "Sheikh",
            "dob": "1993-10-07",
            "mother_name": "Nuzhat",
            "place_of_birth": "Islamabad",
            "account_number": "1005",
            "current_balance_usd": 167952.5,
            "current_balance_pkr": 179325.41
            }

            - bank_statements: { "_id": ObjectId, "account_number": string, "date": ISODate, "type": string ("debit"/"credit"), "description": string, "category": string (e.g., Food, Entertainment), "amount_usd": number, "amount_pkr": number, "balance_usd": number, "balance_pkr": number }

            Example document in bank_statements collection:
            {
            "_id": {
                "$oid": "6874e7bcdfb730a4127a09d8"
            },
            "account_number": "1001",
            "date": {
                "$date": "2025-06-01T00:00:00.000Z"
            },
            "type": "debit",
            "description": "Grocery Store",
            "category": "Groceries",
            "amount_usd": 8468.68,
            "amount_pkr": 0,
            "balance_usd": 37747.04,
            "balance_pkr": 24519.44
            }

            Guidelines:
            - For current balance_inquiry, query the users collection or latest bank_statements document. Set pipeline to [].
            - For transaction_history, use $match, $sort, and optional $limit in the pipeline.
            - For spending_analysis, use $match and $group to aggregate spending by category, description, or date range.
            - For category_spending, use $match and $group for category aggregation.
            - For transfer_money, set pipeline to [] and handle via API.
            - Use ISODate for date filters (e.g., {{"$gte": ISODate("2025-06-01T00:00:00Z")}}).
            - For relative dates (e.g., "last month"), calculate appropriate ISODate ranges based on {current_date}.
            - Ensure the pipeline is valid MongoDB syntax and safe to execute.
            - The account maintains separate USD and PKR balances and transaction amounts. `amount_usd` and `amount_pkr` are independent, as are `balance_usd` and `balance_pkr`. Do not assume any conversion between these fields.

            User query: {user_message}
           ### RESPONSE FORMAT – READ CAREFULLY
            Return **exactly one** valid JSON value that fits the schema above.
            • No Markdown, no ``` fences, no comments, no keys other than the schema.
            • Do not pretty‑print; a single‑line minified object/array is required.
            • If a value is unknown, use null.
            Your entire reply must be parsable by `json.loads`.

            """
        )

intent_prompt = PromptTemplate(
                input_variables=["user_message", "filters"],
                template="""
                You are a banking AI assistant. Analyze the user's query and classify it into one of these intents:

                Available intents:
                1. "balance_inquiry" - User wants to check their account balance
                Examples: "What's my balance?", "How much money do I have?", "Check my account balance", "Show current balance"
                
                2. "transaction_history" - User wants to see their transaction history/list
                Examples: "Show my transactions", "List my recent purchases", "What are my last 10 transactions?", "Transaction history"
                
                3. "spending_analysis" - User wants to analyze their spending on specific items/merchants
                Examples: "How much did I spend on Netflix?", "What did I spend on Amazon last month?", "My Netflix expenses", "How much money went to Uber?"
                
                4. "category_spending" - User wants to analyze spending by category
                Examples: "How much did I spend on food?", "My entertainment expenses", "Food spending last month", "How much on groceries?"
                
                5. "transfer_money" - User wants to transfer money to someone
                Examples: "Transfer money to John", "Send $100 to Alice", "Pay my friend", "Transfer funds"
                
                6. "general" - General queries or unclear intent
                Examples: "Hello", "Help me", "What can you do?", unclear requests

                Classification rules:
                - If user mentions checking balance, money amount, or account status → "balance_inquiry"
                - If user asks for transaction list, history, or recent activities → "transaction_history"
                - If user asks about spending on specific merchants/services (Netflix, Amazon, etc.) → "spending_analysis"
                - If user asks about spending in categories (food, entertainment, etc.) → "category_spending"
                - If user wants to send/transfer money → "transfer_money"
                - If intent is unclear or general → "general"

                Consider these extracted filters to help classification:
                - If filters.limit is set → likely "transaction_history"
                - If filters.description is set → likely "spending_analysis"
                - If filters.category is set → likely "category_spending"
                - If filters.transaction_type is "debit" and specific merchant → likely "spending_analysis"

                User query: "{user_message}"
                Extracted filters: {filters}

                Respond with only the intent name (e.g., "balance_inquiry", "spending_analysis", etc.)
                """
            )

transfer_prompt = PromptTemplate(
                input_variables=["user_message"],
                template="""
                Extract transfer details from the query:
                - amount: number
                - currency: usd/USD = "USD" or pkr/PKR = "PKR" (even if user types currency in lower case always extract it in upper case)
                - recipient: string
                Return JSON: {{"amount": number, "currency": string, "recipient": string}}
                
                Query: {user_message}
                """
            )



# Additional prompts to add to your existing prompts.py file

# Context resolution prompt
context_resolution_prompt = """
You are a banking AI assistant. Resolve this contextual query using the previous conversation context.

Current User Query: "{current_query}"

Previous Context:
- Last Query: "{last_query}"
- Last Intent: "{last_intent}"
- Last Filters: {last_filters}
- Last Response Summary: "{last_response}"

The user is referencing the previous query/result with phrases like "from this", "from that", "out of this", etc.

Your task is to combine the context with the new query to create a complete, standalone query that contains all necessary information.

Examples:
1. Previous: "how much did I spend in June" → Current: "from this how much on groceries" 
   → Resolved: "how much did I spend on groceries in June"

2. Previous: "show me transactions in May" → Current: "from these which ones are over $100"
   → Resolved: "show me transactions in May that are over $100"

3. Previous: "what did I spend on food last month" → Current: "from this breakdown by restaurants vs groceries"
   → Resolved: "breakdown my food spending last month between restaurants and groceries"

Return only the resolved query as a string. Do not include any explanations or formatting.
"""

# Query completeness analysis prompt
completeness_analysis_prompt = """
You are a banking query analyzer. Analyze if this banking query has all necessary information to be processed completely.

Query: "{user_message}"

Check for these requirements based on the query type:

For SPENDING queries ("spend", "spent", "cost", "paid", etc.):
- Time period (month, year, date range, "last week", etc.)
- Category/description (optional but helpful)

For TRANSACTION queries ("show", "list", "transactions", "history"):
- Either a time period OR a limit (e.g., "last 10", "recent 5")

For BALANCE queries:
- Usually complete as-is

For TRANSFER queries:
- Amount, currency, recipient

Common INCOMPLETE examples:
- "how much did I spend on groceries" (missing time period)
- "show me transactions" (missing time period or limit)
- "what did I spend" (missing category and time period)
- "transfer money to John" (missing amount and currency)

Common COMPLETE examples:
- "how much did I spend on groceries in June"
- "show me last 10 transactions"
- "what is my balance"
- "transfer 500 USD to John"

Return JSON:
{{
    "is_complete": true/false,
    "missing_info": ["time_period", "category", "amount", "currency", "recipient", "limit"],
    "clarification_needed": "Specific question to ask user for missing information"
}}

If the query is complete, return {{"is_complete": true}}.
If incomplete, provide a helpful clarification question.
"""

# Intent classification with context prompt
contextual_intent_prompt = """
You are a banking intent classifier. Classify the user's query intent, considering any contextual references.

User Query: "{user_message}"
Extracted Filters: {filters}

Available intents:
1. balance_inquiry - checking account balance
2. transaction_history - viewing past transactions
3. spending_analysis - analyzing spending patterns or totals
4. category_spending - spending breakdown by categories
5. transfer_money - transferring money to someone
6. general - unrecognized or general banking questions

Context indicators:
- If query mentions "from this", "from that", "out of this" etc., it's likely building on a previous query
- Spending queries usually contain words like: spend, spent, cost, paid, expense, purchase
- Transaction queries contain: show, list, history, transactions, recent, last
- Balance queries contain: balance, money, funds, account balance
- Transfer queries contain: transfer, send, pay, wire, move money

Return only the intent name (lowercase, underscore format).

Examples:
- "what is my balance" → balance_inquiry
- "show me last 10 transactions" → transaction_history
- "how much did I spend on food in June" → spending_analysis
- "from this how much on groceries" → spending_analysis (contextual)
- "breakdown my spending by category last month" → category_spending
- "transfer 500 USD to John" → transfer_money
"""