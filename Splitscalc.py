import os, json, sys
from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

from dotenv import load_dotenv
load_dotenv()

# ----- Config -----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
llm = LLM(model="gemini/gemini-2.5-pro", api_key=GEMINI_API_KEY, temperature=0)

# ----- Schema -----
class Row(BaseModel):
    sku: str
    todays_demand: int = Field(ge=0)
    agreed_splits: Dict[str, float]
    actual_splits: Dict[str, float]
    historical_allocations: Dict[str, int]
    supplier_order_multiple: Dict[str, int]
    procurement_calendar: Dict[str, int]  # standardize key

class Payload(BaseModel):
    rows: list[Row]

# ----- Deterministic allocation tool -----
@tool("compute_allocation")
def compute_allocation(payload: Dict[str, Any]) -> str:
    """Deterministically compute per-SKU allocations; returns JSON string."""
    data = Payload(**payload)  # raises if invalid
    out = {}
    for r in data.rows:
        demand = r.todays_demand
        eligible = {s: m for s, m in r.supplier_order_multiple.items()
                    if r.procurement_calendar.get(s, 0) > 0}
        # If no eligible suppliers, allocate zeroes (or raise – business decision)
        if not eligible:
            out[r.sku] = {s: 0 for s in r.supplier_order_multiple.keys()}
            continue

        # Start from agreed splits but only among eligible suppliers
        total_agreed_eligible = sum(r.agreed_splits.get(s, 0.0) for s in eligible)
        if total_agreed_eligible <= 0:
            # fallback to equal split among eligible
            base = {s: 1/len(eligible) for s in eligible}
        else:
            base = {s: r.agreed_splits.get(s, 0.0)/total_agreed_eligible for s in eligible}

        # Under-target weighting (gap = agreed - actual, min 0)
        gaps = {s: max(base.get(s, 0) - r.actual_splits.get(s, 0), 0.0) for s in eligible}
        gap_total = sum(gaps.values())
        pref = ( {s: gaps[s]/gap_total for s in eligible} if gap_total > 0 else base )

        # Raw target units
        raw = {s: demand * pref[s] for s in eligible}

        # Snap to order multiples
        alloc = {}
        remainder = demand
        for s, units in sorted(raw.items(), key=lambda kv: -kv[1]):
            mult = max(eligible[s], 1)
            take = int(units // mult) * mult
            take = min(take, remainder - (len(eligible)-len(alloc)-1))  # keep room
            take = max(take, 0)
            alloc[s] = take
            remainder -= take

        # Distribute remaining units greedily by preference while respecting multiples
        if remainder > 0:
            for s, _ in sorted(pref.items(), key=lambda kv: -kv[1]):
                if remainder <= 0: break
                mult = max(eligible[s], 1)
                add = (remainder // mult) * mult
                if add > 0:
                    alloc[s] += add
                    remainder -= add

        # If still remainder (due to multiples), assign last resort to best-fit supplier
        if remainder > 0:
            best = max(pref.items(), key=lambda kv: kv[1])[0]
            alloc[best] += remainder  # if this violates multiple, you may choose to raise or log
            remainder = 0

        # Non-eligible must be exactly 0
        full = {s: 0 for s in r.supplier_order_multiple}
        full.update(alloc)
        out[r.sku] = full

    return json.dumps(out)

# ----- Tools: load & validate -----
@tool("load_payload")
def load_payload_tool(path: str) -> str:
    """Load payload JSON from a file path and return JSON string."""
    with open(path, "r") as f:
        data = json.load(f)
    # Normalize key from legacy 'Procurement_Calender'
    for row in data.get("rows", []):
        if "procurement_calendar" not in row:
            row["procurement_calendar"] = row.pop("Procurement_Calender", {})
    # Validate early
    Payload(**data)
    return json.dumps(data)

# ----- Agent -----
agent = Agent(
    llm=llm,
    role="Demand Allocation Agent",
    goal="Compute deterministic, constraint-safe allocations and return strict JSON.",
    backstory="You orchestrate tools to load, validate, and compute allocations.",
    tools=[load_payload_tool, compute_allocation],
    allow_delegation=False
)

# ----- Task -----
task = Task(
    description=(
        "1) Use load_payload(path) to read the file path in {path}.\n"
        "2) Use compute_allocation(payload_json) to compute allocations.\n"
        "3) Return ONLY the JSON from compute_allocation with no extra text."
    ),
    expected_output="Strict JSON object of {sku: {supplier: units}}",
    agent=agent
)

# ----- Run -----
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demand_allocator.py <payload.json>"); sys.exit(1)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff(inputs={"path": sys.argv[1]})
    print(result)
