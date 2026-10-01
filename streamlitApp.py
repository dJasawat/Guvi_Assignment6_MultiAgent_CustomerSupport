import os
from groq import Groq
import uuid
import streamlit as st
from Src.graph import graph
from langgraph.types import Command



st.set_page_config(page_title="Customer Support System", page_icon="💬")
st.title("💬  AI powered E- Commerce Customer Support System")
st.caption("Ask a question, analyze its sentiment and Category, and refine the response with an LLM.")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm the support assistant."
				"I am here to help you with your queries"
            ),
        }
    ]

#@st.cache_resource
#def load_sentiment_model(querryTxt):
	#sentiment, confidence = classification_preditor.predict_intent(querryTxt)
	#return sentiment, confidence

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None  # holds interrupt payload while waiting for a human
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_graph(input_state_or_command):
    graph.invoke(input_state_or_command,config=CONFIG)
     # Check whether the graph paused on interrupt() - more robust across
    # LangGraph versions than looking for "__interrupt__" in the invoke result.
    snapshot = graph.get_state(CONFIG)
    if snapshot.next:
        for task in snapshot.tasks:
            if task.interrupts:
                 st.session_state.pending_interrupt = task.interrupts[0].value
                 st.session_state.last_trace = snapshot.values.get("trace",[])
                 return None # nothing final yet waiting for human input

    st.session_state.pending_interrupt = None
    st.session_state.last_trace = snapshot.values.get("trace",[])
    return snapshot.values.get("final_response","(no response generated)")
    
def resume_graph(approved: bool,approver:str,comment:str):
	decision = {"approved": approved,"approver":approver,"comment":comment}
	final_text = run_graph(Command(resume=decision))
	if final_text:
		st.session_state.messages.append({"role":"assistant","content":final_text})



for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

# If the graph is paused on a human-approval interrupt, show the approval card
# instead of (or in addition to) the normal chat input.
if st.session_state.pending_interrupt:
    payload = st.session_state.pending_interrupt
    with st.chat_message("assistant"):
        st.warning("⏸️ This refund needs human approval before it can proceed.")
        st.markdown(
            f"""
**Order:** {payload['order_id']}  &nbsp;|&nbsp; **Customer:** {payload.get('customer_id')}
**Amount:** ${payload['amount']:.2f}  &nbsp;|&nbsp; **Transaction:** {payload['transaction_ref']}
**Policy rule triggered:** `{payload['policy_rule']}`
**Reason:** {payload['reason']}
"""
        )
        with st.form("approval_form", clear_on_submit=True):
            approver = st.text_input("Dimple", value="support_agent")
            comment = st.text_area("Comment (optional, shown to customer if rejected)")
            col1, col2 = st.columns(2)
            approve_clicked = col1.form_submit_button("✅ Approve refund", use_container_width=True)
            reject_clicked = col2.form_submit_button("❌ Reject", use_container_width=True)

        if approve_clicked:
            resume_graph(approved=True, approver=approver, comment=comment)
            st.rerun()
        if reject_clicked:
            resume_graph(approved=False, approver=approver, comment=comment)
            st.rerun()

user_input = st.chat_input("Type your query...",disabled=bool(st.session_state.pending_interrupt))

if user_input:

	st.session_state.messages.append({"role": "user", "content": user_input})
	with st.chat_message("user"):
		st.markdown(user_input)

	initial_state={"ticket_text":user_input,"trace":[]}

	with st.spinner("Processing..."):
    		answer = run_graph(initial_state)
       	 #graph_state = run_graph(initial_state)
        

	# Graph debugging information
	    # Display graph state
   
	#with st.expander("🔍 View Graph State", expanded=False):
	#	st.json(graph_state)

	#answer=	graph_state.values["final_response"]
	if answer:
		st.session_state.messages.append({"role": "assistant", "content": answer})
	st.rerun()
