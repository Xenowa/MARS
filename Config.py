# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATASET_PATH = './Dataset_format/BBH/geometric_shapes.csv'

# Securely get API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Validate that required environment variables are set
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

current_time = None
question_type = None