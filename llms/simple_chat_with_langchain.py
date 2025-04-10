import os
import json
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

# Use env variable directly
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("Error: MISTRAL_API_KEY not found in environment. Please set it in .env.")
    sys.exit(1)

# Custom init_chat_model function based on screenshot
def init_chat_model(model="mistral-large-latest", model_providers=["mistralai"]):
    return ChatMistralAI(
        model_name=model,
        api_key=api_key,
        streaming=True
    )

class ChatManager:
    def __init__(self):
        self.conversation_history = []
        self.chat_model = None
        self.prompt_template = None

    def initialize_context(self):
        try:
            connection_string = os.getenv("MONGODB_URI")
            if not connection_string:
                raise ValueError("MONGODB_URI not set in environment")
            client = MongoClient(connection_string)
            client.server_info()  # Test connection

            db = client['app-dev']
            collection = db['profiles']
            documents = list(collection.find())
            total_count = collection.count_documents({})

            if not documents:
                print("Warning: No profiles found in the 'profiles' collection.")
            else:
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

                large_string = "\n".join(profile_summaries)
                self.conversation_history.append(
                    SystemMessage(content=f"<<< The following is a list of profiles: {large_string} >>>")
                )

            client.close()

            self.chat_model = init_chat_model(model="mistral-large-latest", model_providers=["mistralai"])

            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", "{system_context}"),
                ("human", "{user_input}")
            ])

        except Exception as e:
            print(f"Error initializing context: {e}")
            sys.exit(1)

    async def stream_response(self, prompt):
        if not self.chat_model or not self.prompt_template:
            print("Error: Chat model or prompt template not initialized")
            return

        self.conversation_history.append(HumanMessage(content=prompt))

        messages = self.prompt_template.format_messages(
            system_context=self.conversation_history[0].content,
            user_input=prompt
        )

        print("Response: ", end="", flush=True)
        async for chunk in self.chat_model.astream(messages):
            print(chunk.content, end="", flush=True)
        print()
        self.conversation_history.pop()

if __name__ == "__main__":
    import asyncio

    chat = ChatManager()
    chat.initialize_context()

    print("Enter a prompt (e.g., 'What is John’s expertise?' or 'exit' to quit):")
    while (prompt := input()).lower() != "exit":
        asyncio.run(chat.stream_response(prompt))