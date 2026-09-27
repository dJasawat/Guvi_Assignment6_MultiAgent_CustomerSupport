import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
import Src.llm as llm
#import Src.tools as tools
import Src.db as db


ORDER_ID_REGEX = re.compile(r"\b[A-Z]-\d{3,6}\b")  # Example regex for order IDs like "AB-3456"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SupportState(TypedDict):
    """Represents a state in the graph."""
    name: str
    """The name of the state."""
    ticket_text: str
    order_id: Optional[str]
    intent: Optional[str]
    sentiment: Optional[str]
    faqs:Optional[list[dict]]
    trace: List[str]
    final_response:Optional[str]
    payment_info:Optional[Dict[str, Any]]
    refund_decision: Optional[Dict[str, Any]]
    route: Optional[str]
   

def _log(state: SupportState, message: str) -> None:
    """Logs a message to the state's trace."""
    state.setdefault("trace", []).append(message)

# ---------------------------------------------------------------------------
# Nodes or Agents
# ---------------------------------------------------------------------------

def extract_order_id(ticket_text: str) -> Optional[str]:
    """Extracts an order ID from the ticket text using regex."""
    match = ORDER_ID_REGEX.search(ticket_text)
    return match.group(0) if match else None

def intake_agent(state: SupportState) -> SupportState:
    """Initial node that processes the ticket text and extracts order ID."""
    _log(state, "Intake node: Processing ticket text.")
    order_id = extract_order_id(state["ticket_text"])
    state["order_id"] = order_id

    """Node that classifies the intent of the ticket."""
    intent = llm.classify_intent(state["ticket_text"])
    sentiment = llm.classify_sentiment(state["ticket_text"])
    state["intent"] = intent
    state["sentiment"] = sentiment

    _log(state, f"CLASSIFY :intent={intent}.")
    _log(state, f"CLASSIFY : sentiment={sentiment}")
    _log(state, f"INTAKE: extracted order ID: {order_id}")
    return state


def classify_intent_agent(state: SupportState) -> SupportState:
    """Node that classifies the intent of the ticket."""
    intent = llm.classify_intent(state["ticket_text"])
    state["intent"]= intent
    _log(state, f"CLASSIFY :intent={intent}.")
    return state

# Retrive FAQ database  and past resole Tickets 
def retrieval_agent(state: SupportState) -> SupportState:
    """Node that retrieves payment information based on the order ID."""
    state["faqs"] = db.search_FQA_Qdrant(state["ticket_text"])

    _log(state,f"RETRIVE : FQAS from Qdrant")
    return state


def response_agent(state: SupportState) -> SupportState:
    best_solution = None
    faqs= state["faqs"]
    sentiment = state["sentiment"]
    order_id = state["order_id"]
    intent = state["intent"]

    if faqs and faqs[0]["score"] > 0.15:
        best_solution = faqs[0]["answer"]

    # Build personalized greeting and tone
    if sentiment == "negative":
        greeting = "Hello, we sincerely apologize for the frustration and inconvenience caused."
    elif sentiment == "positive":
        greeting = "Hello! Thanks for reaching out to customer support."
    else:
        greeting = "Hello, thank you for contacting our customer support team."

    order_ref = f" regarding Order {order_id}" if order_id else ""

    if best_solution:
        body = f"In response to your query{order_ref}: {best_solution}"
    else:
        if intent == "Delivery Issue":
            body = f"We are tracking your shipment details{order_ref}. Our carrier team has been notified to provide an urgent status update within 24 hours."
        elif intent == "Refund & Return":
                body = f"We have submitted your return/refund request{order_ref}. Approved refunds are processed to your original payment method within 3 to 5 business days."
        elif intent == "Payment Issue":
                body = f"We are verifying the payment transaction{order_ref}. Any unconfirmed or failed charges will automatically reverse within 24-48 hours."
        elif intent == "Product Issue":
                body = f"We have registered your product concern{order_ref}. A prepaid return shipping label and replacement request has been queued for your account."
        else:
                body = f"Your support ticket{order_ref} has been received and prioritized by our team."

    closing = "If you need further assistance, please reply to this message. We are here to help!"
        
    facts = f"{greeting}\n\n{body}\n\n{closing}"

    _log(state,f"RESPONSE : Query Response using LLM")
    state["final_response"] = llm.draft_response(facts) 
    return state

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():

     builder= StateGraph(SupportState)

     builder.add_node("intake",intake_agent)
     builder.add_node("retrival",retrieval_agent)
     builder.add_node("response_generated",response_agent)

     builder.add_edge(START,"intake")
     builder.add_edge("intake","retrival")
     builder.add_edge("retrival","response_generated")
     builder.add_edge("response_generated",END)
     memory = MemorySaver()
     return builder.compile(checkpointer=memory)

# Singleton graph instance used by the Streamlit app
graph = build_graph()