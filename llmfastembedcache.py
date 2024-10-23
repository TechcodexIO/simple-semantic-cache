"""
Author: Gary Herbst 
Description: Extremely simple python semantic cache, implements an interactive chat interface using OpenAI's GPT model
             with a semantic cache powered by Qdrant vector database for efficient response retrieval.
             This version uses FastEmbed for generating embeddings instead of OpenAI.
References: 
    https://arxiv.org/html/2406.00025v1
    https://arxiv.org/html/2403.02694v1
    https://qdrant.tech
    https://github.com/qdrant/fastembed

Libraries:
    pip install qdrant-client
    pip install openai
    pip install python-dotenv
    pip install fastembed

Qdrant Dashboard:
    http://localhost:6333/dashboard

Instructions:
    1. Run: "docker compose up" in a terminal session to start Qdrant.
    2. Copy the env.example file to .env and edit the .env file with your OpenAI API key and Qdrant connection settings.
    3. In another terminal session, run the python code and interact with the chat interface (python llmfastembedcache.py)
    4. Type 'quit' to end the conversation.
    5. Type 'clear cache' to clear the entire cache for this session.  
"""

import os
import sys
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

# Load environment variables from .env file
load_dotenv()

def check_env_variable(var_name: str, error_message: str) -> None:
    """
    Check if an environment variable is set and exit the program if it's not.

    Args:
        var_name (str): The name of the environment variable to check.
        error_message (str): The error message to display if the variable is not set.

    Raises:
        SystemExit: If the environment variable is not set.
    """
    if not os.getenv(var_name):
        print(f"Error: {error_message}")
        sys.exit(1)

# Check for required environment variables
check_env_variable("OPENAI_API_KEY", "OPENAI_API_KEY environment variable is not set")
check_env_variable("LLMFASTEMBEDCACHE_COLLECTION_NAME", "LLMFASTEMBEDCACHE_COLLECTION_NAME environment variable is not set")

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up GPT model
gpt_model = os.getenv("GPT_MODEL", "gpt-3.5-turbo")

# Set up similarity threshold
similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.95"))

# Set up Qdrant URL, port, and api key
qdrant_url = os.getenv("QDRANT_URL", "http://localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

# Set up FastEmbed
embedding_model = TextEmbedding()

# Set up Qdrant client
try:
    qdrant_client = QdrantClient(
        url=qdrant_url,
        port=qdrant_port,
        api_key=qdrant_api_key
    )
    collection_name = os.getenv("LLMFASTEMBEDCACHE_COLLECTION_NAME")

    # Create collection if it doesn't exist
    try:
        qdrant_client.get_collection(collection_name)
    except UnexpectedResponse:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
except Exception as e:
    print(f"Error connecting to Qdrant: {str(e)}")
    print("Please check your Qdrant connection settings and ensure the service is running (docker compose up).")
    sys.exit(1)

def get_embedding(text: str):
    """
    Generate an embedding for the given text using FastEmbed.

    Args:
        text (str): The input text to generate an embedding for.

    Returns:
        list[float]: A list of floats representing the embedding.
    """
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()

def search_cache(query: str, top_k: int = 5):
    """
    Search the Qdrant vector database for similar queries.

    Args:
        query (str): The query to search for.
        top_k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        list: A list of search results from Qdrant.

    Reference:
        https://qdrant.tech/documentation/quick_start/#search-for-similar-vectors
    """
    query_vector = get_embedding(query)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return search_result

def add_to_cache(query: str, response: str, embedding: list[float]):
    """
    Add a query-response pair and its embedding to the Qdrant cache.

    Args:
        query (str): The user's query.
        response (str): The assistant's response.
        embedding (list[float]): The embedding vector for the query.
    """
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[PointStruct(
            id=str(uuid.uuid4()),  # Generate a UUID as the point ID
            vector=embedding,
            payload={"query": query, "response": response}
        )]
    )

def check_cache(query: str):
    """
    Check if a similar query exists in the cache.

    Args:
        query (str): The user's query.

    Returns:
        tuple[str | None, float | None]: A tuple containing the cached response (if found) and its similarity score.
                                         Returns (None, None) if no similar query is found in the cache.
    """
    results = search_cache(query, top_k=1)
    if results and results[0].score >= similarity_threshold:
        return results[0].payload['response'], results[0].score
    return None, None

def get_openai_response(query: str):
    """
    Get a response from OpenAI's API, using the cache if a similar query exists.

    Args:
        query (str): The user's query.

    Returns:
        str: The assistant's response.

    Reference:
        https://platform.openai.com/docs/api-reference/chat/create
    """
    cached_response, score = check_cache(query)
    if cached_response:
        print(f"Response found in cache. \nProbability: {score:.4f}")
        return cached_response

    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    response_text = response.choices[0].message.content
    
    # Generate embedding for the query
    embedding = get_embedding(query)
    
    # Add the response and embedding to the cache
    add_to_cache(query, response_text, embedding)
    
    return response_text

def clear_cache():
    """
    Clear the entire Qdrant cache by deleting and recreating the collection.

    Reference:
        https://qdrant.tech/documentation/quick_start/#delete-collection
    """
    qdrant_client.delete_collection(collection_name)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("Cache cleared successfully.")

def main():
    """
    Main function to run the interactive chat interface.
    """
    print("Welcome to the Interactive Chat with OpenAI and Semantic Cache (FastEmbed version)")
    print("Type 'quit' to end the conversation.")
    print("Type 'clear cache' to clear the entire cache for this session.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'clear cache':
            clear_cache()
            continue

        try:
            response = get_openai_response(user_input)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please check your environment variables and connection settings.")
            break

if __name__ == "__main__":
    main()
