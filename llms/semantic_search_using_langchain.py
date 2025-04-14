import os
from dotenv import load_dotenv
import sys
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Custom InMemoryVectorStore class with batch embedding
class InMemoryVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = {}  # Dictionary to hold doc_id: (content, embedding) pairs
        self.next_id = 0

    def add_documents(self, documents):
        ids = []
        contents = [doc.page_content for doc in documents]
        # Batch embed all documents
        embeddings = self.embeddings.embed_documents(contents)
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            self.store[self.next_id] = {"content": doc.page_content, "embedding": embedding, "metadata": doc.metadata}
            ids.append(self.next_id)
            self.next_id += 1
        return ids

    def similarity_search(self, query, k=4):
        query_embedding = self.embeddings.embed_query(query)
        # Simple cosine similarity (approximation)
        scores = []
        for doc_id, data in self.store.items():
            score = sum(a * b for a, b in zip(query_embedding, data["embedding"]))
            scores.append((score, doc_id))
        scores.sort(reverse=True)
        top_k = scores[:k]
        return [self._get_document(doc_id) for _, doc_id in top_k]

    def _get_document(self, doc_id):
        data = self.store.get(doc_id, {})
        return type('Document', (), {
            'page_content': data.get('content', ''),
            'metadata': data.get('metadata', {})
        })()

def setup_semantic_search():
    print("Starting setup...", flush=True)

    # Load environment variables from .env
    load_dotenv()

    # Check for Mistral API key
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY not found in .env file. Please set it.", flush=True)
        exit(1)

    # Optional: Check for Hugging Face token
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = "dummy"
    else:
        print("HF token loaded from .env.", flush=True)

    # Step 1: Load the PDF
    file_path = r"C:\Users\user.DESKTOP-OMQ89VA\Desktop\Nike-NPS-Combo_Form-10.pdf"
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.", flush=True)
        exit(1)
    except Exception as e:
        print(f"Error loading PDF: {e}", flush=True)
        exit(1)

    # Step 2: Split the documents into chunks
    print("Splitting documents...", flush=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Created {len(all_splits)} document chunks.", flush=True)

    # Step 3: Create embeddings and store in vector store
    try:
        print("Creating embeddings...", flush=True)
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        print("Embeddings initialized.", flush=True)

        # Test API key with a single embedding
        test_vector = embeddings.embed_query("This is a test sentence.")

        # Test embedding a single document chunk
        single_chunk = all_splits[0].page_content
        single_embedding = embeddings.embed_query(single_chunk)
        assert len(test_vector) == len(single_embedding)  # Verify vector length

        # Create and populate custom in-memory vector store with first 20 chunks
        print("Creating in-memory vector store...", flush=True)
        vector_store = InMemoryVectorStore(embeddings)
        print("In-memory vector store created.", flush=True)

        print("Adding first 20 documents to vector store for testing...", flush=True)
        start_time = time.time()
        ids = vector_store.add_documents(all_splits[:20])  # Add only the first 20 chunks
        elapsed_time = time.time() - start_time
        print(f"Stored {len(ids)} chunks in the vector store. Took {elapsed_time:.2f} seconds.", flush=True)

        # Commented out section for adding all documents
        # print("Adding all documents to vector store...", flush=True)
        # start_time = time.time()
        # ids = vector_store.add_documents(all_splits)
        # elapsed_time = time.time() - start_time
        # print(f"Stored {len(ids)} chunks in the vector store. Took {elapsed_time:.2f} seconds.", flush=True)

    except Exception as e:
        print(f"Error creating embeddings or vector store: {e}", flush=True)
        exit(1)

    # Step 4: Set up the LLM and prompt template
    try:
        print("Setting up LLM...", flush=True)
        llm = ChatMistralAI(model="open-mistral-nemo", temperature=0)
        print("LLM initialized.", flush=True)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant answering questions based on the provided document context. Use only the given context to answer concisely and accurately. If the context doesn't contain the answer, say so.\n\nContext: {context}"),
            ("human", "{question}")
        ])
        chain = prompt_template | llm | StrOutputParser()
        print("LLM chain created.", flush=True)
    except Exception as e:
        print(f"Error setting up LLM: {e}", flush=True)
        exit(1)

    return vector_store, chain

def run_semantic_search(vector_store, chain):
    print("\nWelcome to the Nike PDF Semantic Search with RAG!", flush=True)
    print("Enter a question about the document (or 'exit' to quit).", flush=True)
    while True:
        query = input("Your question: ")
        print(f"Received query: {query}", flush=True)
        if query.lower() == "exit":
            print("Exiting program.", flush=True)
            break

        try:
            print("Performing similarity search...", flush=True)
            results = vector_store.similarity_search(query, k=4)
            if not results:
                print("No relevant information found for your query.", flush=True)
                continue
            print(f"Found {len(results)} results.", flush=True)

            context = "\n\n".join([f"Page {doc.metadata.get('page', 'unknown')}: {doc.page_content}" for doc in results])
            print("Context prepared.", flush=True)

            print("Generating answer with LLM...", flush=True)
            response = chain.invoke({"context": context, "question": query})
            print("Answer generated.", flush=True)

            print("\n--- Answer ---", flush=True)
            print(response, flush=True)
            print("\n--- Source Chunks ---", flush=True)
            for i, doc in enumerate(results, 1):
                print(f"\nChunk {i}:", flush=True)
                print(f"Content: {doc.page_content[:200]}...", flush=True)
                print(f"Source: Page {doc.metadata.get('page', 'unknown')}", flush=True)
        except Exception as e:
            print(f"Error during search or answer generation: {e}", flush=True)

if __name__ == "__main__":
    vector_store, chain = setup_semantic_search()
    run_semantic_search(vector_store, chain)