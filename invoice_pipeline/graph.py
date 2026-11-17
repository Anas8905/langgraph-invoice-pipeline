"""Invoice pipeline: parallel extraction, deterministic validation, bounded repair, quarantine.

Batch graph      load_inbox --Send per file--> process_document (subgraph, in parallel) --> build_report
Document graph   extract --not an invoice--> reject
                    |--> validate --valid--> accept
                            |--invalid, repairs < 2--> repair --changed--> validate
                            |                             |--unchanged--> quarantine
                            |--invalid, repairs = 2--> quarantine
"""
import json
from collections import Counter
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, RetryPolicy, Send

from .state import BatchState, DocOutput, DocResult, DocState, Extraction
from .validate import validate_invoice

MAX_REPAIRS = 2

EXTRACT_PROMPT = """Extract invoice data from the document.
- Copy vendor and invoice_number exactly as printed.
- invoice_date as YYYY-MM-DD; currency as a 3-letter ISO code (£ is GBP, € is EUR, $ is USD
  unless stated otherwise).
- Amounts as plain numbers: "1.204,50" and "1,204.50" both become 1204.50.
- subtotal, tax and total are the values printed on the document. Never compute or correct them.
- Use null for anything not printed."""

REPAIR_PROMPT = """Your previous extraction of this document failed validation.
Re-read the document and fix fields you extracted wrongly.
If the document itself is inconsistent (for example, its line items do not add up to its
printed subtotal), keep the printed values. Never change a number just to make the checks pass."""


def build_document_graph(llm: BaseChatModel):
    extractor = llm.with_structured_output(Extraction)

    def extract(state: DocState) -> dict:
        result = extractor.invoke([("system", EXTRACT_PROMPT),
                                   ("human", f"File: {state['file']}\n\n{state['text']}")])
        return {"extraction": result.model_dump()}

    def validate(state: DocState) -> dict:
        return {"errors": validate_invoice(state["extraction"], state["text"])}

    def repair(state: DocState) -> dict:
        failed = "\n".join(f"- {e}" for e in state["errors"])
        result = extractor.invoke([("system", EXTRACT_PROMPT), ("human",
            f"File: {state['file']}\n\n{state['text']}\n\n{REPAIR_PROMPT}\n\n"
            f"Previous extraction:\n{json.dumps(state['extraction'])}\n\nFailed checks:\n{failed}")])
        fixed = result.model_dump()
        return {"extraction": fixed, "repairs": state["repairs"] + 1,
                "repair_changed": fixed != state["extraction"]}

    def finish(status: str):
        def node(state: DocState) -> dict:
            errors = list(state["errors"])
            if status == "rejected":
                errors.append("not an invoice")
            elif status == "quarantined" and state.get("repair_changed") is False:
                errors.append("repair returned the same values: inconsistent as printed")
            return {"results": [DocResult(file=state["file"], status=status, invoice=state["extraction"],
                                          errors=errors, repairs=state["repairs"])]}
        return node

    def after_extract(s: DocState) -> str:
        return "validate" if s["extraction"]["is_invoice"] else "reject"

    def after_validate(s: DocState) -> str:
        if not s["errors"]:
            return "accept"
        return "repair" if s["repairs"] < MAX_REPAIRS else "quarantine"

    def after_repair(s: DocState) -> str:
        return "validate" if s["repair_changed"] else "quarantine"   # same answer twice: stop paying for it

    g = StateGraph(DocState, output_schema=DocOutput)
    llm_retry = RetryPolicy(max_attempts=3)
    g.add_node("extract", extract, retry_policy=llm_retry)
    g.add_node("validate", validate)
    g.add_node("repair", repair, retry_policy=llm_retry)
    g.add_node("accept", finish("accepted"))
    g.add_node("quarantine", finish("quarantined"))
    g.add_node("reject", finish("rejected"))
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", after_extract, ["validate", "reject"])
    g.add_conditional_edges("validate", after_validate, ["accept", "repair", "quarantine"])
    g.add_conditional_edges("repair", after_repair, ["validate", "quarantine"])
    for terminal in ("accept", "quarantine", "reject"):
        g.add_edge(terminal, END)
    return g.compile()


def make_report(results: list[DocResult]) -> dict:
    """Fan-in. Sorting by file name makes the report independent of parallel completion order.
    Duplicates (same vendor + invoice number) can only be caught here, after every branch is done."""
    documents, seen = [], {}
    for result in sorted(results, key=lambda r: r["file"]):
        result = dict(result)
        if result["status"] == "accepted":
            key = (result["invoice"]["vendor"].casefold(), result["invoice"]["invoice_number"])
            if key in seen:
                result.update(status="quarantined", errors=[f"duplicate of {seen[key]}"])
            else:
                seen[key] = result["file"]
        documents.append(result)
    totals: dict[str, float] = {}
    for d in documents:
        if d["status"] == "accepted":
            currency = d["invoice"]["currency"]
            totals[currency] = round(totals.get(currency, 0.0) + d["invoice"]["total"], 2)
    return {"documents": documents, "counts": dict(Counter(d["status"] for d in documents)),
            "accepted_totals": totals}


def build_graph(llm: BaseChatModel, checkpointer: Checkpointer = None):
    def load_inbox(state: BatchState) -> dict:
        files = sorted(Path(state["inbox"]).glob("*.txt"))
        return {"documents": [{"file": f.name, "text": f.read_text(encoding="utf-8")} for f in files]}

    def fan_out(state: BatchState):
        if not state["documents"]:
            return "build_report"
        return [Send("process_document", {"file": d["file"], "text": d["text"], "extraction": None,
                                          "errors": [], "repairs": 0, "results": []})
                for d in state["documents"]]

    def build_report(state: BatchState) -> dict:
        return {"report": make_report(state["results"])}

    g = StateGraph(BatchState)
    g.add_node("load_inbox", load_inbox)
    g.add_node("process_document", build_document_graph(llm))
    g.add_node("build_report", build_report)
    g.add_edge(START, "load_inbox")
    g.add_conditional_edges("load_inbox", fan_out, ["process_document", "build_report"])
    g.add_edge("process_document", "build_report")
    g.add_edge("build_report", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
