import os
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # or any other model you prefer
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
QA_PAIR_COLLECTION = "FQA_pairs"

client=QdrantClient(
    url= os.environ["QDRANT_URL"],
    #api_key= os.environ["QDRANT_API_KEY"]
    api_key= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Zjc1YjY2ZDItMjNjMi00NTBkLWExMzQtNTRiNGQ0ZDdhNWQ0In0.PpQhBRxf-igLXXPcNxW20cqpCNvIL8j0pgFs6PWisCs"
)

def search_FQA_Qdrant(query: str,top_k: int = 3) -> list[dict]:
    # Embed the query
    query_vector = embedding_model.encode([query])[0].tolist()

    #perform search in QA Pair
    qa_search_result = client.query_points(
        collection_name=QA_PAIR_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True # Include payload in the results
    )
    qa_Formatted_results = []
    #extract relevent infor from the results
    for point in qa_search_result.points:
        payload = point.payload or {}
        qa_Formatted_results.append({
            "score": point.score,
            "category": payload.get("category", ""),
            "answer": payload.get("answer", "")})

    return qa_Formatted_results


