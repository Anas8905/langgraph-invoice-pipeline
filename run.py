"""Process every .txt file in an inbox folder in parallel and print every state transition.

    uv run run.py                     # process ./inbox
    uv run run.py --inbox path/to/dir
    uv run run.py --resume RUN_ID     # finish a batch that crashed; completed documents are not redone
"""
import argparse
import os
import sqlite3
import sys
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from invoice_pipeline.graph import build_graph


def money(invoice: dict) -> str:
    total = invoice.get("total")
    return f"{total:,.2f} {invoice.get('currency')}" if total is not None else "no total"


def describe(node: str, update: dict) -> str:
    if node in ("extract", "repair"):
        ex = update["extraction"]
        summary = (f"{ex['vendor']} | {ex['invoice_number']} | {len(ex['line_items'])} lines | "
                   f"{money(ex)}") if ex["is_invoice"] else "not an invoice"
        return f"attempt {update['repairs']}, changed={update['repair_changed']}: {summary}" \
            if node == "repair" else summary
    if node == "validate":
        errors = update["errors"]
        return "valid" if not errors else f"{len(errors)} failed: " + "; ".join(errors)
    result = update["results"][0]
    return result["status"] + (f" ({'; '.join(result['errors'][-1:])})" if result["errors"] else "")


def run(graph, run_id: str, inputs) -> None:
    config = {"configurable": {"thread_id": run_id}}
    files: dict[str, str] = {}   # parallel task id -> file name
    # "tasks" mode carries each node's full result, including nodes inside the per-document subgraphs
    for ns, task in graph.stream(inputs, config, stream_mode="tasks", subgraphs=True):
        if not ns and task["name"] == "process_document" and "input" in task:
            files[task["id"]] = task["input"]["file"]
        elif "result" not in task or task["name"] == "process_document":
            continue
        elif task["error"]:
            print(f"  {files.get(ns[0].split(':', 1)[1], '') if ns else '':<34} {task['name']:<11} ERROR {task['error']!r}")
        elif ns:   # a node inside one document's subgraph
            file = files.get(ns[0].split(":", 1)[1], "?")
            print(f"  {file:<34} {task['name']:<11} {describe(task['name'], task['result'])}")
        else:
            print(f"── {task['name']}")

    report = graph.get_state(config).values.get("report")
    if not report:
        sys.exit(f"Run {run_id} has no report yet.")
    print(f"\n══ report [{run_id}]")
    for d in report["documents"]:
        inv = d["invoice"] or {}
        what = f"{inv['invoice_number']}  {money(inv)}" if inv.get("is_invoice") else ""
        print(f"  {d['file']:<34} {d['status']:<12} repairs={d['repairs']}  {what}")
        if d["status"] != "accepted":
            print(f"  {'':<34} └ {'; '.join(d['errors'])}")
    print(f"  counts: {report['counts']}   accepted totals: {report['accepted_totals']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inbox", default="inbox")
    parser.add_argument("--resume", metavar="RUN_ID")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    graph = build_graph(llm, SqliteSaver(sqlite3.connect("checkpoints.sqlite", check_same_thread=False)))

    if args.resume:
        return run(graph, args.resume, None)
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    print(f"run {run_id}: processing {args.inbox}/ (resume with --resume {run_id} if it stops)\n")
    run(graph, run_id, {"inbox": args.inbox, "documents": [], "results": [], "report": None})


if __name__ == "__main__":
    main()
