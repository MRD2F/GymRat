from dotenv import load_dotenv
import os
def get_private_keys(key_name):
    load_dotenv()
    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"Missing key for {key_name} in .env") 
    return key

