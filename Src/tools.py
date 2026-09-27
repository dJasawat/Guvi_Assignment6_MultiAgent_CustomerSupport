"""
tools.py
--------
These are the "tools" the agents call. Today they hit the mock CSV
database in db.py. Tomorrow, swap the body of each function for a real
call to a payments MCP server / REST API and nothing else in the graph
needs to change.
"""

from datetime import datetime
import db


def verify_payment(order_id: str) -> dict:
    """
    PAYMENT AGENT tool.
    Looks up the order + all payment attempts for it, and detects
    whether there are two successful charges (a duplicate charge).
    """
    order = db.get_order(order_id)
    if order is None:
        return {"found": False, "order_id": order_id}

    payments = db.get_payments_for_order(order_id)
    succeeded = [p for p in payments if p["status"] == "succeeded"]

    is_duplicate = len(succeeded) >= 2
    original_payment = succeeded[0] if succeeded else None
    duplicate_payment = succeeded[1] if is_duplicate else None

    minutes_apart = None
    if is_duplicate:
        delta = duplicate_payment["charge_date"] - original_payment["charge_date"]
        minutes_apart = round(delta.total_seconds() / 60, 1)

    return {
        "found": True,
        "order_id": order_id,
        "order": order,
        "all_payments": payments,
        "is_duplicate": is_duplicate,
        "original_payment": original_payment,
        "duplicate_payment": duplicate_payment,
        "minutes_apart": minutes_apart,
    }


def check_refund_policy(payment_info: dict) -> dict:
    """
    REFUND AGENT tool.
    Applies the refund policy (data/policy_config.csv) to decide:
      - is this eligible for a refund at all?
      - what amount?
      - can it be auto-approved, or does it need a human?
    """
    if not payment_info.get("found"):
        return {
            "eligible": False, "auto": False, "amount": 0.0,
            "reason": f"Order {payment_info.get('order_id')} was not found in the system.",
            "rule": "R0_NOT_FOUND",
        }

    if not payment_info.get("is_duplicate"):
        return {
            "eligible": False, "auto": False, "amount": 0.0,
            "reason": "Only one successful charge was found for this order - no duplicate charge detected, so this does not qualify for an automatic duplicate-charge refund.",
            "rule": "R4_NO_DUPLICATE",
        }

    dup = payment_info["duplicate_payment"]
    amount = float(dup["amount"])
    charge_date = dup["charge_date"]

    window_days = int(db.get_policy_value("REFUND_WINDOW_DAYS"))
    max_auto = float(db.get_policy_value("AUTO_REFUND_MAX_AMOUNT"))
    dup_window_minutes = float(db.get_policy_value("DUPLICATE_DETECTION_WINDOW_MINUTES"))

    days_since_charge = (db.now() - charge_date).days
    within_window = days_since_charge <= window_days
    within_dup_window = (payment_info.get("minutes_apart") or 0) <= dup_window_minutes

    if not within_dup_window:
        return {
            "eligible": False, "auto": False, "amount": 0.0,
            "reason": f"Two charges exist but they are {payment_info['minutes_apart']} minutes apart, outside the {dup_window_minutes}-minute duplicate-detection window, so they are treated as separate legitimate charges.",
            "rule": "R5_NOT_DUPLICATE_TIMING",
        }

    if not within_window:
        return {
            "eligible": True, "auto": False, "amount": amount,
            "reason": f"Duplicate charge of ${amount:.2f} confirmed, but it is {days_since_charge} days old, outside the {window_days}-day refund window. Needs manual/human review.",
            "rule": "R3_OUTSIDE_WINDOW",
        }

    if amount <= max_auto:
        return {
            "eligible": True, "auto": True, "amount": amount,
            "reason": f"Duplicate charge of ${amount:.2f} confirmed within {window_days}-day window and at/under the ${max_auto:.2f} auto-refund limit. Eligible for automatic refund.",
            "rule": "R1_AUTO_APPROVE",
        }

    return {
        "eligible": True, "auto": False, "amount": amount,
        "reason": f"Duplicate charge of ${amount:.2f} confirmed, but it exceeds the ${max_auto:.2f} auto-refund limit. Needs human approval.",
        "rule": "R2_ABOVE_AUTO_LIMIT",
    }


def execute_refund(order_id: str, payment_id: str, amount: float,
                    method: str = "auto", approved_by: str = "system") -> dict:
    """
    Executes the refund: marks the duplicate payment as refunded and
    writes an audit-log entry. Replace with a real refund API call
    (e.g. Stripe Refunds API / payments MCP tool) in production.
    """
    db.mark_payment_refunded(payment_id)
    refund_id = db.new_refund_id()
    record = {
        "refund_id": refund_id,
        "order_id": order_id,
        "payment_id": payment_id,
        "amount": amount,
        "method": method,
        "approved_by": approved_by,
        "reason": "duplicate_charge_refund",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    db.append_refund_log(record)
    return {"status": "refunded", **record}
