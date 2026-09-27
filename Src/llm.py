"""
llm.py
------
Thin wrapper around the Gemini API (Google), used for two jobs only:
  1. classify_intent()  - read an open-ended customer message, pick an intent and sentiment
  2. draft_response()   - turn a dict of already-decided facts into natural
                           customer-facing language


is never made by the LLM - that stays fully deterministic in tools.py's
policy engine. The LLM only classifies text and writes prose.

--------------------------------------------------------------------
SETUP
1. pip install google-genai
2. Get a free API key at https://aistudio.google.com/apikey
3. Put it in a file named ".env" in this same folder (next to app.py):

       GEMINI_API_KEY=your-key-here

   (There's an ".env.example" file here you can copy/rename.)
--------------------------------------------------------------------

If no key is found, every function here falls back to simple keyword
rules / template text, so the app still runs without a key.
"""

import os
import json
from typing import Optional
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads the ".env" file in this folder, if present
except ImportError:
    pass  # python-dotenv is optional; env vars set another way still work

# Flash-Lite is fast/cheap - good for a short classification call.
CLASSIFY_MODEL = "gemini-3.1-flash-lite"
# Flash gives better prose for the customer-facing message.
DRAFT_MODEL = "openai/gpt-oss-120b"

GROQ_API = os.environ.get("GROQ_KEY")

"""
    Cleans text and extracts Intent, Sentiment, Urgency, and Entities (Order ID, Product Name, Amount).
    """

NEGATIVE_WORDS = {
        "angry", "furious", "terrible", "awful", "horrible", "frustrated", "disappointed",
        "broken", "damaged", "cracked", "defective", "worst", "ignored", "late", "stolen",
        "wrong", "twice", "failed", "unacceptable", "scam", "poor"
    }

POSITIVE_WORDS = {
        "thanks", "thank", "great", "good", "happy", "love", "awesome", "excellent",
        "please", "kindly", "inquire", "warranty", "loyalty", "redeem", "schedule"
    }
URGENT_WORDS = {
        "immediately", "urgent", "asap", "furious", "angry", "ignored", "overdue", "stolen"
    }

VALID_INTENTS = {"Refund & Return", "Delivery Issue", "Product Issue","Payment Issue","Account & Login","Others"}

VALID_SENTIMENTS = {"positive", "neutral", "negative"}

CLASSIFY_INTENT_SYSTEM_PROMPT = """You are an intent classifier for an e-commerce support inbox.
Classify the customer's message into exactly one category:
- Refund & Return: they say they were charged more than once / double-charged / duplicate charge for the same order
- Delivery Issue: they have a problem with their delivery
- Product Issue: they have a problem with the product they received
- Payment Issue: they have a problem with their payment
- Account & Login: they have issues with their account or login
- Others: anything else (question, complaint, shipping issue, etc. that isn't a refund request)

Respond with ONLY compact JSON, nothing else, in this exact shape:
{"intent": "Refund & Return"}"""

CLASSIFY_SENTIMENT_SYSTEM_PROMPT = """You are a sentiment classifier for an e-commerce support inbox.
Classify the customer's message into exactly one category:
- positive: query is polite, friendly, or neutral
- neutral : query is neither positive nor negative
- negative : query is angry, frustrated, or upset

Respond with ONLY compact JSON, nothing else, in this exact shape:
{"sentiment": "negative"}"""

DRAFT_SYSTEM_PROMPT = """You are a warm, concise customer support agent for an online store.
You will be given facts that were already decided by the
company's policy system (outcome, order id, amount, refund id, reason, etc.).
Write a short customer-facing reply (3-5 sentences) based ONLY on the facts.
Never invent, guess, or add amounts, IDs, or policy details not present in the facts.
Do not over-apologize. Be direct, friendly, and clear about what happens next."""

_client = None
_client_checked = False

_groqclient = Groq(api_key=GROQ_API.strip())
def _get_client():
    """Returns a cached Gemini client, or None if no API key is configured."""
    global _client, _client_checked
    if not _client_checked:
        _client_checked = True
        api_key = os.environ.get("GEMINI_KEY") 
        if api_key:          
            from google import genai
            _client = genai.Client(api_key=api_key)
    return _client


def classify_intent(text: str) -> str:
    """
    Classifies a support ticket's intent using Gemini.
    Falls back to keyword rules if no API key is set or the call fails,
    so the graph still works offline / without a key.
    """
    client = _get_client()
    if client is not None:
        try:
            from google.genai import types
            resp = client.models.generate_content(
                model=CLASSIFY_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_INTENT_SYSTEM_PROMPT,
                    max_output_tokens=50,
                ),
            )
            data = json.loads(resp.text.strip())
            intent = data.get("intent")
            if intent in VALID_INTENTS:
                return intent
        except Exception as e:
            print(f"[llm.classify_intent] falling back to rules - {e}")

    return _rule_based_intent_classify(text)

def classify_sentiment(text: str) -> str:
    """
    Classifies a support ticket's sentiment using Gemini.
    Falls back to keyword rules if no API key is set or the call fails,
    so the graph still works offline / without a key.
    """
    client = _get_client()
    if client is not None:
        try:
            from google.genai import types
            resp = client.models.generate_content(
                model=CLASSIFY_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_SENTIMENT_SYSTEM_PROMPT,
                    max_output_tokens=50,
                ),
            )
            data = json.loads(resp.text.strip())
            sentiment = data.get("sentiment")
            if sentiment in VALID_SENTIMENTS:
                return sentiment
        except Exception as e:
            print(f"[llm.classify_sentiment] falling back to rules - {e}")

    return _rule_based_sentiment_classify(text)

def _rule_based_intent_classify(text: str) -> str:
      def extract_intent(self, text: str) -> str:
        """Derives primary user intent based on keywords."""
        t = text.lower()
        if any(k in t for k in ["refund", "return", "money back", "cancel"]):
            return "Refund & Return"
        elif any(k in t for k in ["where is", "track", "delivery", "late", "arrived", "shipping"]):
            return "Delivery Issue"
        elif any(k in t for k in ["charged", "payment", "bill", "invoice", "double"]):
            return "Payment / Invoice Inquiry"
        elif any(k in t for k in ["broken", "damaged", "defective", "cracked", "warranty", "quality"]):
            return "Product Issue"
        else:
            return "General Support Inquiry"

def _rule_based_sentiment_classify(text: str) -> str:
    t = text.lower()
    if any(k in t for k in NEGATIVE_WORDS):
        return "negative"
    elif any(k in t for k in POSITIVE_WORDS):
        return "positive"
    else:
        return "neutral"


def draft_response(facts: str) -> Optional[str]:
    """
    Turns already-decided facts into natural language using Gemini.
    Returns None if no API key is configured or the call fails, so
    callers can fall back to a plain template string.
    """
    #client = _get_client()
    if _groqclient is None:
        return None
    try:
        from google.genai import types

        resp = _groqclient.chat.completions.create(
                 model=DRAFT_MODEL,
                 messages=[{
                            "role":"system",
                            "content": DRAFT_SYSTEM_PROMPT
                         },
                         {
                            "role":"user",
                            "content":json.dumps(facts)
                        }
                     ],
                    max_tokens=400
                )
        return  resp.choices[0].message.content
    except Exception as e:
        print(f"[llm.draft_response] falling back to template - {e}")
        return None


def active_provider_label() -> str:
    """For display in the Streamlit sidebar."""
    client = _get_client()
    return "Gemini (Google)" if client is not None else "No API key found - using rule-based fallback"
