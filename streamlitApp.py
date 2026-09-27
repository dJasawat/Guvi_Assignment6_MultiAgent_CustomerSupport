import os
from groq import Groq
import uuid
import streamlit as st
from Src.graph import graph


st.set_page_config(page_title="Sentiment Chat", page_icon="💬")
st.title("💬 Sentiment-Aware Chat")
st.caption("Ask a question, analyze its sentiment, and refine the response with an LLM.")

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
                "Hi! I'm the support assistant. Tell me what happened - "
				"I am here to help you with your Queries"
            ),
        }
    ]

#@st.cache_resource
#def load_sentiment_model(querryTxt):
	#sentiment, confidence = classification_preditor.predict_intent(querryTxt)
	#return sentiment, confidence


CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_graph(input_state_or_command):
	graph.invoke(input_state_or_command,config=CONFIG)
	 # Check whether the graph paused on interrupt() - more robust across
    # LangGraph versions than looking for "__interrupt__" in the invoke result.
	snapshot= graph.get_state(CONFIG)
	st.markdown("Snapshot values")
	st.markdown(snapshot.values)
	st.session_state.pending_interrupt = None
	st.session_state.last_trace = snapshot.values.get("trace", [])
	#return snapshot.values.get("final_response", "(no response generated)")
	return snapshot
	


for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])


user_input = st.chat_input("Type your query...")

if user_input:

	st.session_state.messages.append({"role": "user", "content": user_input})
	with st.chat_message("user"):
		st.markdown(user_input)

	initial_state={"ticket_text":user_input,"trace":[]}

	with st.spinner("Processing..."):
       	 graph_state = run_graph(initial_state)

	# Graph debugging information
	with st.expander("🔍 View Graph State", expanded=False):
		st.json(graph_state)
	
	answer=	graph_state.values["final_response"]
	if answer:
		st.session_state.messages.append({"role": "assistant", "content": answer})
	st.rerun()
