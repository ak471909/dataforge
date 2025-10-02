# main.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key is loaded
api_key = os.getenv('TDB_OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found! Check your .env file")

print("✓ Environment loaded successfully")