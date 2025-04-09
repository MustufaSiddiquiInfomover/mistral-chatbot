import os
import json
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

def stream_mistral_response(api_key, model, prompt):
    """Stream Mistral API response directly to console"""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "My name is Mustufa Siddiqui"},
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }

    print("Response: ", end="", flush=True)
    with requests.post(url, json=payload, headers=headers, stream=True) as resp:
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} - {resp.text}")
            return
        for line in resp.iter_lines():
            if line and line != b"data: [DONE]":
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:].decode("utf-8"))
                        if content := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
    print()

if __name__ == "__main__":
    if not (api_key := os.getenv("MISTRAL_API_KEY")):
        print("Error: MISTRAL_API_KEY not set")
        sys.exit(1)

    model = "open-mistral-nemo"
    while (prompt := input("Enter prompt ('exit' to quit): ")).lower() != "exit":
        stream_mistral_response(api_key, model, prompt)