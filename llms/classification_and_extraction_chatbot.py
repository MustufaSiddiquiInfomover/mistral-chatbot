import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
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

# Define the ProductReview model for extraction
class ProductReview(BaseModel):
    """Information about a product review."""
    product_name: Optional[str] = Field(default=None, description="The name of the product mentioned in the review")
    rating: Optional[int] = Field(default=None, description="The star rating given in the review, from 1 to 5")
    complaint: Optional[str] = Field(default=None, description="A brief description of the complaint, if any")

def classify_sentences():
    """Classify the given sentences using the language model."""
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY not found in .env file. Please set it.", flush=True)
        exit(1)

    # Create the tagging prompt for classification
    tagging_prompt = ChatPromptTemplate.from_template(
        """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
    )

    # Initialize the model for classification
    llm = ChatMistralAI(model="open-mistral-nemo", temperature=0).with_structured_output(Classification)

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

def extract_review_info():
    """Extract product review information from text using the language model."""
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY not found in .env file. Please set it.", flush=True)
        exit(1)

    # Define the extraction prompt
    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )

    # Initialize the model for extraction
    llm = ChatMistralAI(model="open-mistral-nemo", temperature=0).with_structured_output(ProductReview)

    # Text to extract information from
    review_text = "The TurboBlender 3000 broke after one use, and I’m giving it a 1-star rating because it’s terrible."

    # Perform extraction
    prompt = extraction_prompt.invoke({"text": review_text})
    response = llm.invoke(prompt)
    print(f"\nReview Text: {review_text}")
    print("Extracted Information:", response.model_dump())

def main():
    """Main function to run the classification and extraction program."""
    print("Running Classification Task:")
    classify_sentences()
    print("\nRunning Extraction Task:")
    extract_review_info()

if __name__ == "__main__":
    main()