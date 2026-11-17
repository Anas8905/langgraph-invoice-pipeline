"""Typed state: one for the batch (parent graph), one per document (subgraph)."""
import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class Extraction(BaseModel):
    """What the LLM must return for each document."""
    is_invoice: bool = Field(description="True for a bill requesting payment, including a copy or reminder "
                                         "that contains the full invoice. False for delivery notes, quotes, etc.")
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None = Field(description="YYYY-MM-DD")
    currency: str | None = Field(description="ISO 4217 code, e.g. USD")
    line_items: list[LineItem]
    subtotal: float | None
    tax: float | None
    total: float | None


class DocResult(TypedDict):
    file: str
    status: Literal["accepted", "quarantined", "rejected"]
    invoice: dict | None
    errors: list[str]
    repairs: int


class DocState(TypedDict):
    file: str
    text: str
    extraction: dict | None
    errors: list[str]          # validation errors for the current extraction
    repairs: int
    repair_changed: bool       # did the last repair change anything?
    results: Annotated[list[DocResult], operator.add]


class DocOutput(TypedDict):
    """Only this crosses back into the parent graph."""
    results: Annotated[list[DocResult], operator.add]


class BatchState(TypedDict):
    inbox: str
    documents: list[dict]                               # [{"file", "text"}]
    results: Annotated[list[DocResult], operator.add]   # fan-in: each document appends one result
    report: dict | None
