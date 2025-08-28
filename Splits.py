import os
import sys
import json
from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

from dotenv import load_dotenv
load_dotenv()

# ----- Config -----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY is missing. Put it in .env as GEMINI_API_KEY=your_key")
    sys.exit(1)

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
def compute_allocation(payload_json: str) -> str:
    """Deterministically compute per-SKU allocations; returns JSON string.
       NOTE: Tools receive strings. Parse payload_json first."""
    try:
        payload: Dict[str, Any] = json.loads(payload_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON passed to compute_allocation: {e}"})

    try:
        data = Payload(**payload)  # raises if invalid
    except ValidationError as e:
        return json.dumps({"error": f"Payload validation failed", "details": e.errors()})

    out: Dict[str, Dict[str, int]] = {}
    for r in data.rows:
        demand = r.todays_demand
        eligible = {s: m for s, m in r.supplier_order_multiple.items()
                    if r.procurement_calendar.get(s, 0) > 0}

        # If no eligible suppliers, allocate zeroes
        if not eligible:
            out[r.sku] = {s: 0 for s in r.supplier_order_multiple.keys()}
            continue

        # Start from agreed splits among eligible suppliers
        total_agreed_eligible = sum(r.agreed_splits.get(s, 0.0) for s in eligible)
        if total_agreed_eligible <= 0:
            base = {s: 1/len(eligible) for s in eligible}
        else:
            base = {s: r.agreed_splits.get(s, 0.0)/total_agreed_eligible for s in eligible}

        # Under-target weighting (gap = agreed - actual, min 0)
        gaps = {s: max(base.get(s, 0) - r.actual_splits.get(s, 0), 0.0) for s in eligible}
        gap_total = sum(gaps.values())
        pref = ({s: gaps[s]/gap_total for s in eligible} if gap_total > 0 else base)

        # Raw target units
        raw = {s: demand * pref[s] for s in eligible}

        # Snap to order multiples
        alloc: Dict[str, int] = {}
        remainder = demand
        for s, units in sorted(raw.items(), key=lambda kv: -kv[1]):
            mult = max(eligible[s], 1)
            take = int(units // mult) * mult
            # keep room for at least 0 for remaining suppliers
            take = min(take, max(0, remainder - (len(eligible)-len(alloc)-1)))
            take = max(take, 0)
            alloc[s] = take
            remainder -= take

        # Distribute remaining units by preference while respecting multiples
        if remainder > 0:
            for s, _ in sorted(pref.items(), key=lambda kv: -kv[1]):
                if remainder <= 0:
                    break
                mult = max(eligible[s], 1)
                add = (remainder // mult) * mult
                if add > 0:
                    alloc[s] += add
                    remainder -= add

        # Last resort: if still remainder (e.g., multiples too large), give to best pref
        if remainder > 0:
            best = max(pref.items(), key=lambda kv: kv[1])[0]
            alloc[best] += remainder
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
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": f"Payload file not found: {path}"})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in payload file: {e}"})

    # Normalize legacy key 'Procurement_Calender' -> 'procurement_calendar'
    for row in data.get("rows", []):
        if "procurement_calendar" not in row:
            row["procurement_calendar"] = row.pop("Procurement_Calender", {})

    # Validate early
    try:
        Payload(**data)
    except ValidationError as e:
        return json.dumps({"error": "Payload validation failed", "details": e.errors()})

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
        "You must output ONLY raw JSON with no extra text.\n"
        "Steps:\n"
        f"1) Call load_payload with the exact argument {{path}} to read the file path.\n"
        "2) Pass the JSON string returned by load_payload directly to compute_allocation.\n"
        "3) Return ONLY the JSON returned by compute_allocation."
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

    # Print the tool/LLM output reliably
    if hasattr(result, "raw") and result.raw:
        print(result.raw)
    elif hasattr(result, "tasks_output") and result.tasks_output:
        for t in result.tasks_output:
            print(getattr(t, "raw", None) or getattr(t, "result", None) or str(t))
    else:
        print(result)
