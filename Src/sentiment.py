import re
from typing import Dict, Any

class IntakeNLPProcessor:
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

    def clean_text(self, text: str) -> str:
        """Sanitizes text by removing extra spaces, special chars while retaining order IDs."""
        if not text:
            return ""
        text = text.strip()
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extracts Order IDs,  amounts, and potential product names."""
        entities = {}
        
        # Order ID pattern: #ORD-12345 or ORD-12345 or order 12345
        order_match = re.search(r'#?(ORD|ord|Order|order)[-_\s]?\d{4,6}', text)
        if order_match:
            entities["order_id"] = order_match.group(0).upper().replace(" ", "-")
        else:
            entities["order_id"] = None

        # Amount pattern: $49.99 or 49.99 dollars
        amount_match = re.search(r'\$\d+(\.\d{1,2})?', text)
        if amount_match:
            entities["amount"] = amount_match.group(0)
        else:
            entities["amount"] = None
       
        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Calculates sentiment, sentiment score (-1 to 1), and urgency status."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        urgent_count = sum(1 for w in words if w in self.URGENT_WORDS)

        total_match = neg_count + pos_count
        if neg_count > pos_count:
            sentiment = "Negative"
            score = -min(1.0, 0.4 + 0.2 * neg_count)
        elif pos_count > neg_count:
            sentiment = "Positive"
            score = min(1.0, 0.4 + 0.2 * pos_count)
        else:
            sentiment = "Neutral"
            score = 0.0

        is_angry = neg_count >= 2 or "angry" in text_lower or "furious" in text_lower
        is_urgent = urgent_count > 0 or is_angry

        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "is_angry": is_angry,
            "is_urgent": is_urgent
        }

    def extract_intent(self, text: str) -> str:
        """Derives primary user intent based on keywords."""
        t = text.lower()
        if any(k in t for k in ["refund", "return", "money back", "cancel"]):
            return "Request Refund / Return"
        elif any(k in t for k in ["where is", "track", "delivery", "late", "arrived", "shipping"]):
            return "Track Package / Delivery Status"
        elif any(k in t for k in ["charged", "payment", "bill", "invoice", "double"]):
            return "Payment / Invoice Inquiry"
        elif any(k in t for k in ["broken", "damaged", "defective", "cracked", "warranty", "quality"]):
            return "Report Damaged or Defective Item"
        else:
            return "General Support Inquiry"

    def process(self, text: str) -> Dict[str, Any]:
        cleaned = self.clean_text(text)
        sentiment_data = self.analyze_sentiment(cleaned)
        entities = self.extract_entities(cleaned)
        intent = self.extract_intent(cleaned)

        return {
            "cleaned_text": cleaned,
            "intent": intent,
            "sentiment": sentiment_data["sentiment"],
            "sentiment_score": sentiment_data["score"],
            "is_angry": sentiment_data["is_angry"],
            "is_urgent": sentiment_data["is_urgent"],
            "entities": entities
        }
