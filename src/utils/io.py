def get_private_key(key_name : str) -> str:
    from dotenv import load_dotenv
    import os

    load_dotenv()
    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"Missing key for {key_name} in .env") 
    return key
