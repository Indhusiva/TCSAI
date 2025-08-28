import os, sys, json
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

# -------- LLM config --------

llm = LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434", temperature=0)

print('llm connected')

# -------- Schema --------
class Row(BaseModel):
    sku: str
    todays_demand: int = Field(ge=0)
    agreed_splits: Dict[str, float]
    actual_splits: Dict[str, float]
    historical_allocations: Dict[str, int]
    supplier_order_multiple: Dict[str, int]
    procurement_calendar: Dict[str, int]

class Payload(BaseModel):
    rows: list[Row]

# -------- Core deterministic compute (pure Python) --------
def _compute_core(payload_dict: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    data = Payload(**payload_dict)
    out: Dict[str, Dict[str, int]] = {}

    for r in data.rows:
        demand = r.todays_demand
        print(demand)
        eligible = {s: m for s, m in r.supplier_order_multiple.items()
                    if r.procurement_calendar.get(s, 0) > 0}

        if not eligible:
            out[r.sku] = {s: 0 for s in r.supplier_order_multiple.keys()}
            continue

        total_agreed_eligible = sum(r.agreed_splits.get(s, 0.0) for s in eligible)
        if total_agreed_eligible <= 0:
            base = {s: 1/len(eligible) for s in eligible}
        else:
            base = {s: r.agreed_splits.get(s, 0.0)/total_agreed_eligible for s in eligible}

        gaps = {s: max(base.get(s, 0) - r.actual_splits.get(s, 0), 0.0) for s in eligible}
        gap_total = sum(gaps.values())
        pref = ({s: gaps[s]/gap_total for s in eligible} if gap_total > 0 else base)

        raw = {s: demand * pref[s] for s in eligible}

        alloc: Dict[str, int] = {}
        remainder = demand
        for s, units in sorted(raw.items(), key=lambda kv: -kv[1]):
            mult = max(eligible[s], 1)
            take = int(units // mult) * mult
            take = min(take, max(0, remainder - (len(eligible)-len(alloc)-1)))
            take = max(take, 0)
            alloc[s] = take
            remainder -= take

        if remainder > 0:
            for s, _ in sorted(pref.items(), key=lambda kv: -kv[1]):
                if remainder <= 0:
                    break
                mult = max(eligible[s], 1)
                add = (remainder // mult) * mult
                if add > 0:
                    alloc[s] += add
                    remainder -= add

        if remainder > 0:
            best = max(pref.items(), key=lambda kv: kv[1])[0]
            alloc[best] += remainder
            remainder = 0

        full = {s: 0 for s in r.supplier_order_multiple}
        full.update(alloc)
        out[r.sku] = full
    print('out', out)
    return out

# -------- Tools (string-in / string-out) --------
@tool("load_payload")
def load_payload_tool(path: str) -> str:
    """Load the payload from a JSON file path, normalize keys, validate schema, and return JSON string."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": f"Payload file not found: {path}"})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in payload file: {e}"})

    # Normalize legacy key
    for row in data.get("rows", []):
        if "procurement_calendar" not in row:
            row["procurement_calendar"] = row.pop("Procurement_Calender", {})

    try:
        Payload(**data)
    except ValidationError as e:
        return json.dumps({"error": "Payload validation failed", "details": e.errors()})
    return json.dumps(data)

@tool("compute_allocation")
def compute_allocation(payload_json: str) -> str:
    """Compute per-SKU allocations deterministically from a payload JSON string; returns allocations JSON string."""
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON passed to compute_allocation: {e}"})
    try:
        out = _compute_core(payload)
        return json.dumps(out)
    except ValidationError as e:
        return json.dumps({"error": "Payload validation failed", "details": e.errors()})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"})

@tool("validate_allocation")
def validate_allocation(args_json: str) -> str:
    """Validate allocations against business rules. Input JSON: {"payload": {...}, "allocations": {...}}.
    Checks per-SKU: (1) total allocation == todays_demand, (2) order multiples respected,
    (3) no allocation to suppliers with procurement_calendar == 0, (4) no negatives.
    Returns JSON: {"ok": true} or {"ok": false, "issues": [{"sku": "...", "reason": "..."}]}."""
    try:
        args = json.loads(args_json)
        payload = args["payload"]
        allocations = args["allocations"]
    except Exception as e:
        return json.dumps({"ok": False, "issues": [{"sku": "_all_", "reason": f"Bad validator input: {e}"}]})

    try:
        data = Payload(**payload)
    except ValidationError as e:
        return json.dumps({"ok": False, "issues": [{"sku": "_all_", "reason": f"Payload invalid: {e.errors()}"}]})

    issues = []
    rows_by_sku = {r.sku: r for r in data.rows}

    for sku, alloc_map in allocations.items():
        if sku not in rows_by_sku:
            issues.append({"sku": sku, "reason": "SKU not present in payload"})
            continue
        r = rows_by_sku[sku]

        # (1) sum equals demand
        total = sum(int(v) for v in alloc_map.values())
        if total != r.todays_demand:
            issues.append({"sku": sku, "reason": f"Sum {total} != todays_demand {r.todays_demand}"})

        # (2) order multiples & (4) non-negative
        for sup, units in alloc_map.items():
            u = int(units)
            if u < 0:
                issues.append({"sku": sku, "reason": f"Negative units for {sup}"})
            mult = r.supplier_order_multiple.get(sup)
            if mult is not None and mult > 0 and u % mult != 0 and u != 0:
                issues.append({"sku": sku, "reason": f"{sup} allocation {u} not multiple of {mult}"})

        # (3) no allocation when calendar == 0
        for sup, cal in r.procurement_calendar.items():
            if cal <= 0:
                if int(alloc_map.get(sup, 0)) != 0:
                    issues.append({"sku": sku, "reason": f"Allocated to ineligible supplier {sup} (calendar=0)"} )

    ok = len(issues) == 0
    return json.dumps({"ok": ok, "issues": issues})

# -------- Agents --------
allocator = Agent(
    llm=llm,
    role="Planner & Allocator",
    goal="Plan tool use to load payload, compute allocations, validate them, and output strict JSON.",
    backstory="A meticulous operations analyst who orchestrates tools and returns only valid JSON.",
    tools=[load_payload_tool, compute_allocation, validate_allocation],
    allow_delegation=False
)

auditor = Agent(
    llm=llm,
    role="Auditor",
    goal="Double-check that the returned JSON is the final clean JSON with no extra text.",
    backstory="Quality gate ensuring outputs are strictly JSON and nothing else.",
    tools=[],  # no tools; just a formatting guard
    allow_delegation=False
)

# -------- Tasks --------
compute_and_validate = Task(
    description=(
        "You must output ONLY raw JSON with no extra text.\n"
        "Plan:\n"
        "1) Call load_payload with {path} to get the payload JSON string.\n"
        "2) Pass that JSON to compute_allocation to get allocations JSON.\n"
        "3) Call validate_allocation with {\"payload\": <payload_dict>, \"allocations\": <alloc_dict>} as JSON string.\n"
        "4) If validation is not ok, you MUST call compute_allocation again (optionally adjust your internal plan), "
        "then re-validate until ok.\n"
        "5) Return ONLY the final allocations JSON (strict JSON object of {sku: {supplier: units}})."
    ),
    expected_output="Strict JSON object of {sku: {supplier: units}}",
    agent=allocator
)

finalize_output = Task(
    description=(
        "Ensure the previous task's output is ONLY a JSON object with no commentary. "
        "If any non-JSON text exists, extract and return only the JSON object."
    ),
    expected_output="Strict JSON object of {sku: {supplier: units}}",
    agent=auditor
)

# -------- Run --------
def _extract_first_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        starts = [i for i, ch in enumerate(text) if ch == "{"]
        for i in starts:
            for j in range(len(text)-1, i, -1):
                if text[j] == "}":
                    try:
                        return json.loads(text[i:j+1])
                    except Exception:
                        continue
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demand_allocator.py <payload.json>")
        sys.exit(1)

    crew = Crew(agents=[allocator, auditor], tasks=[compute_and_validate, finalize_output], verbose=False)
    result = crew.kickoff(inputs={"path": sys.argv[1]})
    print('test')
    # Normalize & print clean JSON only
    text = (getattr(result, "raw", None)
            or getattr(result, "output", None)
            or ("\n".join([(getattr(t, "raw", None) or getattr(t, "result", None) or str(t))
                           for t in getattr(result, "tasks_output", [])]))
            or str(result))

    parsed = _extract_first_json(text if isinstance(text, str) else str(text))
    if parsed is not None:
        print(json.dumps(parsed, indent=2))
    else:
        print(text)  # fallback so you can see what happened
