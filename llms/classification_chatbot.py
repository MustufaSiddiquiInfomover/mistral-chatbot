import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

# Define the Classification model
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )

def classify_sentences():
    """Classify the given sentences using the language model."""
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY not found in .env file. Please set it.", flush=True)
        exit(1)

    # Create the tagging prompt
    tagging_prompt = ChatPromptTemplate.from_template(
        """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
    )

    llm = ChatMistralAI(model="open-mistral-nemo",temperature=0).with_structured_output(Classification)
    # Sentences to classify
    sentences = [
        "Hola amigo, ¿cómo estás hoy?",
        "¡Estoy completamente harto y furioso, este servicio de entrega es una absoluta desgracia!"
    ]

    for sentence in sentences:
        prompt = tagging_prompt.invoke({"input": sentence})
        response = llm.invoke(prompt)
        print(f"\nSentence: {sentence}")
        print("Classification:", response.model_dump())

def main():
    """Main function to run the classification program."""
    classify_sentences()

if __name__ == "__main__":
    main()