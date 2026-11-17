"""Offline tests: the real inbox, a scripted LLM keyed by file, every branch of the subgraph."""
import random
import sqlite3
import threading
from datetime import date

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from invoice_pipeline.graph import build_graph, make_report
from invoice_pipeline.state import Extraction, LineItem
from invoice_pipeline.validate import printed, validate_invoice


def inv(vendor, number, day, currency, lines, subtotal, tax, total):
    return Extraction(is_invoice=True, vendor=vendor, invoice_number=number, invoice_date=day, currency=currency,
                      line_items=[LineItem(description=d, quantity=q, unit_price=p, amount=a) for d, q, p, a in lines],
                      subtotal=subtotal, tax=tax, total=total)


NOT_INVOICE = Extraction(is_invoice=False, vendor=None, invoice_number=None, invoice_date=None, currency=None,
                         line_items=[], subtotal=None, tax=None, total=None)
NORTHWIND = inv("Northwind Office Supply", "INV-1043", "2026-10-02", "USD",
                [("Printer paper", 40, 18.5, 740.0), ("Toner", 4, 85.0, 340.0), ("Organiser", 5, 20.0, 100.0)],
                1180.0, 94.4, 1274.4)
HARBOR = inv("Harbor Print Co.", "7781", "2026-10-14", "GBP",
             [("Brochure print run", 1, 520.0, 520.0), ("Business cards", 4, 30.0, 120.0)], 640.0, 128.0, 768.0)
RIDGEWAY = inv("Ridgeway Logistics", "RL-2207", "2026-10-09", "EUR",
               [("Freight", 1, 1450.0, 1450.0), ("Fuel", 1, 300.0, 300.0), ("Handling", 4, 100.0, 400.0)],
               2250.0, 225.0, 2475.0)
ALPINE = inv("Alpine Cloud GmbH", "AC-0099", "2026-10-03", "EUR",
             [("Compute instances", 1, 1000.0, 1000.0), ("Object storage", 1, 204.5, 204.5)], 1204.5, 228.86, 1433.36)

FAITHFUL = {"northwind-INV-1043.txt": [NORTHWIND], "reminder-northwind-INV-1043.txt": [NORTHWIND],
            "harbor-print-7781.txt": [HARBOR], "ridgeway-RL-2207.txt": [RIDGEWAY],
            "alpine-cloud-AC-0099.txt": [ALPINE], "delivery-note-5512.txt": [NOT_INVOICE]}


class ScriptedLLM:
    """Replies per file, in order; the last reply repeats. Exceptions in a script are raised."""
    def __init__(self, script):
        self.script = {f: list(replies) for f, replies in script.items()}
        self.calls: dict[str, int] = {}
        self.lock = threading.Lock()

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        file = messages[1][1].split("\n", 1)[0].removeprefix("File: ")
        with self.lock:
            self.calls[file] = self.calls.get(file, 0) + 1
            replies = self.script[file]
            reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def run_batch(llm, checkpointer=None, inbox="inbox", thread="t1"):
    graph = build_graph(llm, checkpointer)
    config = {"configurable": {"thread_id": thread}}
    graph.invoke({"inbox": inbox, "documents": [], "results": [], "report": None}, config)
    report = graph.get_state(config).values["report"]
    return {d["file"]: d for d in report["documents"]}, report


# ── the inbox end to end ──────────────────────────────────────────────────────────────

def test_inbox_is_sorted_into_accepted_quarantined_rejected():
    docs, report = run_batch(ScriptedLLM(FAITHFUL))
    status = {f: d["status"] for f, d in docs.items()}
    assert status == {"alpine-cloud-AC-0099.txt": "accepted", "harbor-print-7781.txt": "accepted",
                      "northwind-INV-1043.txt": "accepted", "delivery-note-5512.txt": "rejected",
                      "reminder-northwind-INV-1043.txt": "quarantined", "ridgeway-RL-2207.txt": "quarantined"}
    assert docs["reminder-northwind-INV-1043.txt"]["errors"] == ["duplicate of northwind-INV-1043.txt"]
    assert "line items sum to 2,150.00 but subtotal is 2,250.00" in docs["ridgeway-RL-2207.txt"]["errors"]
    assert report["accepted_totals"] == {"EUR": 1433.36, "GBP": 768.0, "USD": 1274.4}


def test_repair_fixes_a_misread_field():
    took_net_as_total = HARBOR.model_copy(update={"total": 640.0})
    llm = ScriptedLLM({**FAITHFUL, "harbor-print-7781.txt": [took_net_as_total, HARBOR]})
    docs, _ = run_batch(llm)
    assert docs["harbor-print-7781.txt"]["status"] == "accepted"
    assert docs["harbor-print-7781.txt"]["repairs"] == 1 and llm.calls["harbor-print-7781.txt"] == 2


def test_inconsistent_invoice_stops_after_an_unchanged_repair():
    llm = ScriptedLLM(FAITHFUL)
    docs, _ = run_batch(llm)
    ridgeway = docs["ridgeway-RL-2207.txt"]
    assert ridgeway["repairs"] == 1 and llm.calls["ridgeway-RL-2207.txt"] == 2   # not 3: early exit
    assert "inconsistent as printed" in ridgeway["errors"][-1]


def test_repair_cannot_invent_numbers_that_add_up():
    fudged = RIDGEWAY.model_copy(update={"subtotal": 2150.0, "total": 2375.0})  # arithmetic now "works"
    llm = ScriptedLLM({**FAITHFUL, "ridgeway-RL-2207.txt": [RIDGEWAY, fudged, RIDGEWAY]})
    docs, _ = run_batch(llm)
    ridgeway = docs["ridgeway-RL-2207.txt"]
    assert ridgeway["status"] == "quarantined" and ridgeway["repairs"] == 2


def test_fudged_numbers_fail_provenance_check():
    fudged = RIDGEWAY.model_copy(update={"subtotal": 2150.0, "total": 2375.0}).model_dump()
    text = open("inbox/ridgeway-RL-2207.txt", encoding="utf-8").read()
    errors = validate_invoice(fudged, text)
    assert "subtotal 2,150.00 is not printed in the document" in errors
    assert "total 2,375.00 is not printed in the document" in errors


def test_empty_inbox_still_produces_a_report(tmp_path):
    docs, report = run_batch(ScriptedLLM({}), inbox=str(tmp_path))
    assert docs == {} and report["counts"] == {}


def test_report_does_not_depend_on_completion_order():
    docs, _ = run_batch(ScriptedLLM(FAITHFUL))
    results = list(docs.values())
    for _ in range(5):
        random.shuffle(results)
        fresh = [{**r, "status": "accepted", "errors": []} if r["file"].startswith("reminder") else r for r in results]
        report = make_report(fresh)
        dup = next(d for d in report["documents"] if d["file"].startswith("reminder"))
        assert dup["status"] == "quarantined"


# ── failure handling: partial batch failure resumes from checkpoint ───────────────────

def test_crashed_batch_resumes_without_redoing_finished_documents():
    saver, config = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)), {"configurable": {"thread_id": "b"}}
    broken = ScriptedLLM({**FAITHFUL, "alpine-cloud-AC-0099.txt": [RuntimeError("model unavailable")]})
    graph = build_graph(broken, saver)
    with pytest.raises(RuntimeError):
        graph.invoke({"inbox": "inbox", "documents": [], "results": [], "report": None}, config)

    healthy = ScriptedLLM(FAITHFUL)
    final = build_graph(healthy, saver).invoke(None, config)
    assert len(final["report"]["documents"]) == 6
    assert healthy.calls == {"alpine-cloud-AC-0099.txt": 1}   # finished branches were checkpointed


# ── validator ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,text,expected", [
    (1204.5, "Nettobetrag 1.204,50 €", True),
    (1204.5, "Subtotal 1,204.50", True),
    (204.5, "Nettobetrag 1.204,50 €", False),     # not a match inside a larger number
    (768.0, "Amount due £768.00", True),
    (76.8, "Amount due £768.00", False),
])
def test_printed(value, text, expected):
    assert printed(value, text) is expected


def test_validator_flags_invented_fields():
    text = open("inbox/northwind-INV-1043.txt", encoding="utf-8").read()
    bad = NORTHWIND.model_copy(update={"invoice_number": "INV-1034", "currency": "usd",
                                       "invoice_date": "2027-01-01", "vendor": "Northwind Ltd"}).model_dump()
    errors = validate_invoice(bad, text, today=date(2026, 11, 17))
    assert any("INV-1034" in e for e in errors) and any("ISO 4217" in e for e in errors)
    assert any("future" in e for e in errors) and any("Northwind Ltd" in e for e in errors)
    assert validate_invoice(NORTHWIND.model_dump(), text) == []
