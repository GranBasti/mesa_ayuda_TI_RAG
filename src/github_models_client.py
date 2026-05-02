import os
import requests
from dotenv import load_dotenv

load_dotenv()

def ask_github_model(system_prompt: str, user_prompt: str) -> str:
    token = os.getenv("GITHUB_TOKEN")
    endpoint = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.github.ai/inference/chat/completions")
    model = os.getenv("GITHUB_MODEL", "openai/gpt-4.1")
    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("MAX_TOKENS", "350"))

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]