import os
import json
import subprocess
import sys
from mistralai import Mistral
from dotenv import load_dotenv

# First, ensure we have the requests library
try:
    import requests
except ImportError:
    print("Installing requests library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

load_dotenv()

def get_user_input():
    print("Enter your prompt (type 'exit' to quit):")
    user_input = input()
    return user_input

def stream_mistral_response(api_key, model, user_input):
    """Stream response using direct API requests"""
    api_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "My name is Mustufa Siddiqui"
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "stream": True
    }

    print("Response: ", end="", flush=True)
    with requests.post(api_url, json=payload, headers=headers, stream=True) as response:
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode('utf-8')

            # Skip the [DONE] message
            if line_text == "data: [DONE]":
                break

            # Process data lines
            if line_text.startswith('data: '):
                data_str = line_text[6:]  # Remove 'data: ' prefix

                try:
                    data = json.loads(data_str)
                    if 'choices' in data and data['choices']:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content')
                        if content:
                            print(content, end='', flush=True)
                except json.JSONDecodeError:
                    pass

    print()  # End with a new line

if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    model = "open-mistral-nemo"

    while True:
        user_input = get_user_input()
        if user_input.lower() == 'exit':
            break

        # Use direct HTTP streaming which is more reliable
        stream_mistral_response(api_key, model, user_input)