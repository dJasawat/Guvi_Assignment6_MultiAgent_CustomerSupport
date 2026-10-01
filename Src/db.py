import os
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import uuid

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "Data"
)
DATA_DIR = os.path.abspath(DATA_DIR)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # or any other model you prefer
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
QA_PAIR_COLLECTION = "FQA_pairs"

CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.csv")
ORDERS_PATH = os.path.join(DATA_DIR, "orders.csv")
PAYMENTS_PATH = os.path.join(DATA_DIR, "payments.csv")
POLICY_CONFIG_PATH = os.path.join(DATA_DIR, "policy_config.csv")
REFUNDS_LOG_PATH = os.path.join(DATA_DIR, "refunds_log.csv")

# ---------------------------------------------------------------------------
# Load once at import time (kept in memory for the life of the process).
# ---------------------------------------------------------------------------
_customers = pd.read_csv(CUSTOMERS_PATH)
_orders = pd.read_csv(ORDERS_PATH)
_payments = pd.read_csv(PAYMENTS_PATH, parse_dates=["charge_date"])
_policy_config = pd.read_csv(POLICY_CONFIG_PATH)

# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------
def get_customer(customer_id: str):
    row = _customers[_customers["customer_id"] == customer_id]
    return None if row.empty else row.iloc[0].to_dict()


def get_order(order_id: str):
    row = _orders[_orders["order_id"] == order_id]
    return None if row.empty else row.iloc[0].to_dict()


def get_payments_for_order(order_id: str):
    rows = _payments[_payments["order_id"] == order_id].sort_values("charge_date")
    records = [r.to_dict() for _, r in rows.iterrows()]
    # Convert pandas Timestamp -> native datetime so LangGraph's checkpoint
    # serializer (used for the human-in-the-loop interrupt) can handle it.
    for rec in records:
        if hasattr(rec["charge_date"], "to_pydatetime"):
            rec["charge_date"] = rec["charge_date"].to_pydatetime()
    return records

def get_policy_value(key: str):
    row = _policy_config[_policy_config["key"] == key]
    if row.empty:
        raise KeyError(f"Unknown policy key: {key}")
    return row.iloc[0]["value"]

# get FAQ from qdrant 
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


# ---------------------------------------------------------------------------
# Write helpers (mutate in-memory tables + append-only audit log on disk)
# ---------------------------------------------------------------------------
def mark_payment_refunded(payment_id: str):
    global _payments
    _payments.loc[_payments["payment_id"] == payment_id, "status"] = "refunded"


def append_refund_log(record: dict):
    df = pd.DataFrame([record])
    df.to_csv(REFUNDS_LOG_PATH, mode="a", header=False, index=False)


def new_refund_id() -> str:
    return f"RF-{uuid.uuid4().hex[:8].upper()}"


def now() -> datetime:
    return datetime.now()