"""Wire contract for a trusted tool host. IDs and sequence numbers route events only."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FIELDS = (
    "tool",
    "operation",
    "classification",
    "destination",
    "scope",
    "permission",
    "state",
)
WINDOW = 12
MAX_HISTORY = 1024  # terminal events one request may carry


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1]
    session: str = Field(min_length=1, max_length=96)
    tenant: str = Field(min_length=1, max_length=96)
    seq: int = Field(ge=1, strict=True)
    tool: Literal["search", "read", "summarize", "stage", "submit"]
    operation: Literal[
        "lookup", "amend", "fetch", "compose", "case_summary", "bulletin", "upload"
    ]
    classification: Literal["catalog", "case", "restricted", "draft"]
    destination: Literal["local", "case_portal"]
    scope: Literal["case", "related_lookup", "outside"]
    permission: Literal["allow", "deny"]
    state: Literal["proposed", "completed", "denied", "failed", "held"]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposed: Event
    history: list[Event] = Field(max_length=MAX_HISTORY)

    @model_validator(mode="after")
    def check_stream(self):
        p = self.proposed
        if p.state != "proposed" or p.seq % 2 != 1:
            raise ValueError("expected an odd-numbered proposal without an outcome")
        last = 0
        for h in self.history:
            if (h.tenant, h.session) != (p.tenant, p.session):
                raise ValueError("history crosses a session or tenant boundary")
            if h.seq <= last or h.seq >= p.seq or h.seq % 2 or h.state == "proposed":
                raise ValueError("expected ordered prior terminal events")
            if h.permission == "deny" and h.state != "denied":
                raise ValueError("inconsistent permission and outcome")
            last = h.seq
        return self

    @property
    def complete(self):
        # Ordering and bounds are already validated. Any gap reduces this count.
        return len(self.history) == self.proposed.seq // 2
