import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
endpoint = os.getenv("GITHUB_MODELS_ENDPOINT")
model = os.getenv("GITHUB_MODEL", "openai/gpt-4.1")

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json"
}

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Responde solo: conexión correcta"}
    ],
    "temperature": 0.2,
    "max_tokens": 30
}

response = requests.post(endpoint, headers=headers, json=payload)
print("STATUS:", response.status_code)
print(response.text)