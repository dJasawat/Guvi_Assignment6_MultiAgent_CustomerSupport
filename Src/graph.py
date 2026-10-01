import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
import Src.llm as llm
import Src.tools as tools
import Src.db as db
import Src.agentLogs as logs


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
    priorty: Optional[str]
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
    # save state in a file
    logs.logger.info(message)



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

def intent_selector(state: SupportState) -> str:
    return "Payment" if state["intent"] in ("Refund & Return", "Payment Issue") else "other"

def route_selector(state: SupportState) -> str:
    return state["route"]

def payment_agent(state: SupportState) -> SupportState:
    order_id = state.get("order_id")
    if not order_id:
        state["payment_info"] = {"found": False, "order_id": None}
        _log(state, "PAYMENT AGENT: no order_id found in ticket")
        return state

    info = tools.verify_payment(order_id)
    state["payment_info"] = info
    _log(
        state,
        f"PAYMENT AGENT: found={info['found']} "
        f"is_duplicate={info.get('is_duplicate')} "
        f"minutes_apart={info.get('minutes_apart')}",
    )
    return state

def refund_agent(state: SupportState) -> SupportState:
    decision = tools.check_refund_policy(state["payment_info"])
    state["refund_decision"] = decision
    _log(
        state,
        f"REFUND AGENT: eligible={decision['eligible']} auto={decision['auto']} "
        f"amount={decision['amount']} rule={decision['rule']}",
    )
    return state

def decide_route(state: SupportState) -> SupportState:
    decision = state["refund_decision"]
    if not decision["eligible"]:
        state["route"] = "reject"
    elif decision["auto"]:
        state["route"] = "auto"
    else:
        state["route"] = "human"
    _log(state, f"ROUTE: -> {state['route']}")
    return state
def auto_refund_agent(state: SupportState) -> SupportState:
    decision = state["refund_decision"]
    payment = state["payment_info"]["duplicate_payment"]
    result = tools.execute_refund(
        order_id=state["order_id"],
        payment_id=payment["payment_id"],
        amount=decision["amount"],
        method="auto",
        approved_by="system",
    )
    _log(state, f"AUTO REFUND: executed {result['refund_id']} for ${result['amount']:.2f}")

    facts = {
        "outcome": "auto_refund_approved",
        "order_id": state["order_id"],
        "amount": result["amount"],
        "refund_id": result["refund_id"],
        "transaction_ref": payment["transaction_ref"],
        "eta": "5-7 business days",
    }
    state["final_response"] = llm.draft_response(facts) or (
        f"Good news - I verified the duplicate charge on order {state['order_id']} "
        f"and automatically refunded ${result['amount']:.2f} "
        f"(refund ID {result['refund_id']}, transaction {payment['transaction_ref']}). "
        f"It should appear on your original payment method within 5-7 business days."
    )
    return state

def human_review_agent(state: SupportState) -> SupportState:
    decision = state["refund_decision"]
    payment_info = state["payment_info"]
    dup = payment_info["duplicate_payment"]

    payload = {
        "type": "refund_approval_request",
        "order_id": state["order_id"],
        "customer_id": state.get("customer_id"),
        "amount": decision["amount"],
        "payment_id": dup["payment_id"],
        "transaction_ref": dup["transaction_ref"],
        "reason": decision["reason"],
        "policy_rule": decision["rule"],
    }
    _log(state, "HUMAN REVIEW: interrupting graph, waiting for approval...")

    # This pauses the graph run. Streamlit resumes it later with
    # Command(resume={"approved": bool, "approver": str, "comment": str})
    human_decision = interrupt(payload)

    state["human_decision"] = human_decision
    _log(state, f"HUMAN REVIEW: resumed with decision={human_decision}")

    if human_decision.get("approved"):
        result = tools.execute_refund(
            order_id=state["order_id"],
            payment_id=dup["payment_id"],
            amount=decision["amount"],
            method="human_approved",
            approved_by=human_decision.get("approver", "support_agent"),
        )
        facts = {
            "outcome": "human_approved_refund",
            "order_id": state["order_id"],
            "amount": result["amount"],
            "refund_id": result["refund_id"],
            "eta": "5-7 business days",
        }
        state["final_response"] = llm.draft_response(facts) or (
            f"Thanks for your patience - a support specialist reviewed order {state['order_id']} "
            f"and approved a refund of ${result['amount']:.2f} "
            f"(refund ID {result['refund_id']}). It will post to your original payment method "
            f"within 5-7 business days."
        )
    else:
        comment = human_decision.get("comment", "").strip()
        facts = {
            "outcome": "human_rejected_refund",
            "order_id": state["order_id"],
            "reviewer_comment": comment or None,
        }
        state["final_response"] = llm.draft_response(facts) or (
            f"A support specialist reviewed order {state['order_id']} and was not able to approve "
            f"an automatic refund at this time."
            + (f" Note: {comment}" if comment else "")
            + " A team member will follow up with you directly."
        )
    return state

def reject_agent(state: SupportState) -> SupportState:
    decision = state["refund_decision"]
    _log(state, "REJECT: not eligible for refund")
    facts = {
        "outcome": "refund_rejected",
        "order_id": state.get("order_id"),
        "reason": decision["reason"],
    }
    state["final_response"] = llm.draft_response(facts) or (
        f"I looked into order {state.get('order_id')}: {decision['reason']} "
        "If you believe this is incorrect, I can pass this to a support specialist for a closer look."
    )
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
     builder.add_node("retrieval",retrieval_agent)
     builder.add_node("payment_agent",payment_agent)
     builder.add_node("refund_agent",refund_agent)
     builder.add_node("decide_route",decide_route)
     builder.add_node("human_review",human_review_agent)
     builder.add_node("auto_refund_agent",auto_refund_agent)
     builder.add_node("reject_agent",reject_agent)
     builder.add_node("response_generated",response_agent)

     builder.add_edge(START,"intake")
     builder.add_edge("intake","retrieval")
     builder.add_conditional_edges("retrieval",intent_selector,
                                   {"Payment":"payment_agent","other":"response_generated"})

     builder.add_edge("payment_agent","refund_agent")
     builder.add_edge("refund_agent","decide_route")

     builder.add_conditional_edges("decide_route" , route_selector,
        {"auto": "auto_refund_agent", "human": "human_review", "reject": "reject_agent"},
    )
   #  builder.add_edge("retrieval","response_generated")
     
    #builder.add_edge("intake","retrival")
     builder.add_edge("human_review", END)
     builder.add_edge("reject_agent",END)
     builder.add_edge("auto_refund_agent",END)
    
     builder.add_edge("response_generated",END)
     memory = MemorySaver()
     return builder.compile(checkpointer=memory)

# Singleton graph instance used by the Streamlit app
graph = build_graph()