"""Deterministic invoice validation. No LLM.

Two kinds of check:
  arithmetic: quantity x unit price = amount, lines sum to subtotal, subtotal + tax = total
  provenance: every amount, the invoice number and the vendor must be printed in the document

The provenance checks stop the repair loop from "fixing" an inconsistent invoice by inventing
numbers that add up. Arithmetic alone would reward exactly that.
"""
import re
from datetime import date

TOLERANCE = 0.01


def money_forms(value: float) -> set[str]:
    """1204.5 -> {"1,204.50", "1204.50", "1.204,50", "1204,50"}"""
    us = f"{value:,.2f}"
    eu = us.replace(",", "_").replace(".", ",").replace("_", ".")
    return {us, us.replace(",", ""), eu, eu.replace(".", "")}


def printed(value: float, text: str) -> bool:
    return any(re.search(rf"(?<![\d.,]){re.escape(form)}(?!\d|[.,]\d)", text) for form in money_forms(value))


def validate_invoice(ex: dict, text: str, today: date | None = None) -> list[str]:
    errors = [f"{f} is missing" for f in ("vendor", "invoice_number", "invoice_date", "currency", "total")
              if ex.get(f) in (None, "")]
    if ex.get("invoice_number") and not re.search(rf"(?<![\w-]){re.escape(ex['invoice_number'])}(?![\w-])", text):
        errors.append(f"invoice_number {ex['invoice_number']!r} is not printed in the document")
    if ex.get("vendor") and ex["vendor"].lower() not in text.lower():
        errors.append(f"vendor {ex['vendor']!r} is not printed in the document")
    if ex.get("currency") and not re.fullmatch(r"[A-Z]{3}", ex["currency"]):
        errors.append(f"currency {ex['currency']!r} is not an ISO 4217 code")
    if ex.get("invoice_date"):
        try:
            if date.fromisoformat(ex["invoice_date"]) > (today or date.today()):
                errors.append(f"invoice_date {ex['invoice_date']} is in the future")
        except ValueError:
            errors.append(f"invoice_date {ex['invoice_date']!r} is not YYYY-MM-DD")

    items = ex.get("line_items") or []
    for i, item in enumerate(items, 1):
        if abs(item["quantity"] * item["unit_price"] - item["amount"]) > TOLERANCE:
            errors.append(f"line {i}: {item['quantity']:g} x {item['unit_price']:,.2f} is not {item['amount']:,.2f}")
        if not printed(item["amount"], text):
            errors.append(f"line {i}: amount {item['amount']:,.2f} is not printed in the document")
    subtotal, tax, total = ex.get("subtotal"), ex.get("tax"), ex.get("total")
    if items and subtotal is not None and abs(sum(i["amount"] for i in items) - subtotal) > TOLERANCE:
        errors.append(f"line items sum to {sum(i['amount'] for i in items):,.2f} but subtotal is {subtotal:,.2f}")
    if None not in (subtotal, tax, total) and abs(subtotal + tax - total) > TOLERANCE:
        errors.append(f"subtotal {subtotal:,.2f} + tax {tax:,.2f} is not total {total:,.2f}")
    for name, value in (("subtotal", subtotal), ("tax", tax), ("total", total)):
        if value is not None and not printed(value, text):
            errors.append(f"{name} {value:,.2f} is not printed in the document")
    return errors
