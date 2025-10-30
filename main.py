# from dotenv import load_dotenv
# import os

# load_dotenv()  # loads environment variables from .env


from dotenv import load_dotenv
import sys, os

load_dotenv()

# Append src/ manually to sys.path (safe, one-liner)
sys.path.append(os.getenv("PYTHONPATH"))

from utils import get_private_key

