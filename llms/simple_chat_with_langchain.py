import os, json, sys
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY") or sys.exit("Error: MISTRAL_API_KEY not found")

class ChatManager:
    def __init__(self):
        self.conversation_history = []
        self.chat_model = None
        self.prompt_template = None

    def initialize_context(self):
        try:
            client = MongoClient(os.getenv("MONGODB_URI") or sys.exit("Error: MONGODB_URI not set"))
            client.server_info()
            profiles = client['app-dev']['profiles'].find()

            summaries = [
                f"Name: {doc.get('firstName', '')} {doc.get('lastName', '')}\n"
                f"Slug: {doc.get('slug', '')}\n"
                f"Expertise: {doc.get('areaOfExpertise', '')}\n"
                f"Type: {doc.get('type', '')}\n"
                f"Current Location: {doc.get('currentLocation', '')}\n"
                f"Career Summary: {doc.get('careerSummary', '')}\n"
                for doc in profiles
            ]

            if not summaries:
                print("Warning: No profiles found")
            else:
                profile_text = "\n".join(summaries)
                self.conversation_history.append(SystemMessage(content=f"<<< Profiles: {profile_text} >>>"))

            client.close()

            self.chat_model = ChatMistralAI(model_name="mistral-large-latest", api_key=api_key, streaming=True)
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", "{system_context}"),
                ("human", "{user_input}")
            ])

        except Exception as e:
            sys.exit(f"Error initializing: {e}")

    async def stream_response(self, prompt):
        if not (self.chat_model and self.prompt_template):
            return print("Error: Model or template not initialized")

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
    while (prompt := input("Enter prompt ('exit' to quit): ")).lower() != "exit":
        asyncio.run(chat.stream_response(prompt))