import os
from dotenv import load_dotenv

load_dotenv()
print("TOKEN cargado:", bool(os.getenv("GITHUB_TOKEN")))
print("MODEL:", os.getenv("GITHUB_MODEL"))