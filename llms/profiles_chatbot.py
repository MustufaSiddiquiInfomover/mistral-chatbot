import os
import json
import sys
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

class ChatManager:
    def __init__(self):
        self.conversation_history = []

    def initialize_context(self):
        """
        Get all profiles data from the database
        Create a readable string from each row and concatenate into a single context
        Add a new object to the conversation_history
        """
        try:
            connection_string = os.getenv("MONGODB_URI", "mongodb+srv://hiretalent-dev:Yulwbmn87x92EQ0U@hiretalent.doscksq.mongodb.net/app-dev")
            client = MongoClient(connection_string)

            # Test connection
            client.server_info()  # Raises exception if connection fails
            print("Successfully connected to MongoDB!")

            db = client['app-dev']
            collection = db['profiles']

            # Fetch all profiles without limit
            documents = list(collection.find())
            total_count = collection.count_documents({})  # Verify total profiles
            print(f"Total profiles in database: {total_count}")
            print(f"Fetched {len(documents)} profiles.")

            if not documents:
                print("Warning: No profiles found in the 'profiles' collection.")
            else:
                # Create a readable string from profile data
                profile_summaries = []
                for doc in documents:
                    summary = (
                        f"Name: {doc.get('firstName', '')} {doc.get('lastName', '')}\n"
                        f"Slug: {doc.get('slug', '')}\n"
                        f"Expertise: {doc.get('areaOfExpertise', '')}\n"
                        f"Type: {doc.get('type', '')}\n"
                        f"Current Location: {doc.get('currentLocation', '')}\n"
                        f"Career Summary: {doc.get('careerSummary', '')}\n"
                    )
                    profile_summaries.append(summary)

                # Join summaries into a single context string
                large_string = "\n".join(profile_summaries)
                self.conversation_history.append({
                    "role": "system",
                    "content": f"<<< The following is a list of profiles: {large_string} >>>"
                })

            client.close()
        except Exception as e:
            print(f"Error initializing context: {e}")

    def stream_mistral_response(self, api_key, model, prompt):
        """Stream Mistral API response directly to console"""
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": self.conversation_history + [{"role": "user", "content": prompt}],
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
    # Initialize the chat manager and load context
    chat = ChatManager()
    chat.initialize_context()

    if not (api_key := os.getenv("MISTRAL_API_KEY")):
        print("Error: MISTRAL_API_KEY not set")
        sys.exit(1)

    model = "open-mistral-nemo"
    print("Enter a prompt (e.g., 'What is John’s expertise?' or 'exit' to quit):")
    while (prompt := input()).lower() != "exit":
        chat.stream_mistral_response(api_key, model, prompt)