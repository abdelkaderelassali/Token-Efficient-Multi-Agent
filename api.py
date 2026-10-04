import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pathlib import Path
import threading
from typing import Literal

from benchmark import build_multi_agent_graph, initial_state
from graph_verified_lossless_workflow import MODEL as PAIRED_MODEL, run_pair as run_lossless_pair, scenarios as demo_scenarios
from graph_verified_lossless_workflow import PROTOCOL as LOSSLESS_PROTOCOL
from graph_selective_workflow import run_pair as run_selective_pair
from graph_selective_workflow import PROTOCOL as SELECTIVE_PROTOCOL
from adaptive_workflow import run_pair as run_adaptive_pair, PROTOCOL as ADAPTIVE_PROTOCOL
from custom_scenarios import normalize_custom, FIELDS as CUSTOM_FIELDS
from requirement_review import assess_requirements, blocked_result
from validation import validate_report

app = FastAPI()
ROOT = Path(__file__).resolve().parent
comparison_lock = threading.Lock()


class CompareRequest(BaseModel):
    scenario_id: str
    compression_mode: Literal["auto", "always", "never"] = "auto"
    fidelity: Literal["complete", "decision", "adaptive"] = "complete"
    custom_scenario: dict | None = None


def paired_protocol(fidelity: str) -> str:
    return {'complete': LOSSLESS_PROTOCOL, 'decision': SELECTIVE_PROTOCOL,
            'adaptive': ADAPTIVE_PROTOCOL}[fidelity]


@app.get('/api/evaluation/latest')
def latest_evaluation(fidelity: Literal["complete", "decision", "adaptive"] = "complete"):
    results = []
    for path in (ROOT / 'verification').glob('*.summary.json'):
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            if result.get('protocol') == paired_protocol(fidelity) and result.get('phase') in {'pilot', 'repeat_check', 'benchmark', 'evaluate', 'stress'}:
                results.append(result)
        except (ValueError, OSError, AttributeError):
            continue
    if not results:
        raise HTTPException(status_code=404, detail='No saved model results for the current protocol yet')
    phase_rank = {'stress': 0, 'pilot': 1, 'repeat_check': 2, 'benchmark': 3, 'evaluate': 4}
    return max(results, key=lambda r: (phase_rank[r['phase']], r['created_at']))


@app.post("/api/compare")
def compare_scenario(request: CompareRequest):
    if request.scenario_id == 'custom':
        try:
            scenario = normalize_custom(request.custom_scenario)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        if request.custom_scenario is not None:
            raise HTTPException(status_code=422, detail='Custom data requires scenario_id=custom')
        scenario = next((s for s in demo_scenarios() if s["id"] == request.scenario_id), None)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Unknown paired scenario")
    if not comparison_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A paired experiment is already running. Try again after it finishes.")
    try:
        runner = {'complete': run_lossless_pair, 'decision': run_selective_pair,
                  'adaptive': run_adaptive_pair}[request.fidelity]
        review = assess_requirements(scenario)
        if scenario.get('requirement_review', {}).get('required_capabilities') and request.fidelity != 'adaptive':
            review['status'] = 'needs_review'
            review['issues'].append('Select Adaptive mode to validate additional supplier capabilities.')
        if review['status'] != 'ready':
            result = blocked_result(scenario, review, paired_protocol(request.fidelity), PAIRED_MODEL)
        else:
            result = runner(scenario, compression_mode=request.compression_mode)
            result['requirement_validation'] = review
        destination = ROOT / "verification" / f"paired-{result['experiment_id']}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps([result], indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Comparison failed: {exc}") from exc
    finally:
        comparison_lock.release()


@app.get("/api/compare/latest/{scenario_id}")
def latest_comparison(scenario_id: str, model: str = PAIRED_MODEL,
                      fidelity: Literal["complete", "decision", "adaptive"] = "complete"):
    import re
    if scenario_id not in {s["id"] for s in demo_scenarios()} and not re.fullmatch(r'custom_[a-f0-9]{24}', scenario_id):
        raise HTTPException(status_code=404, detail="Unknown paired scenario")
    matches = []
    # Live comparisons and CLI evaluations both persist arrays of experiments.
    for path in (ROOT / "verification").glob("*.json"):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                continue
            matches.extend(r for r in records if isinstance(r, dict)
                           and r.get("schema_version") == 1
                           and r.get("configuration", {}).get("pipeline_version") == paired_protocol(fidelity)
                           and r.get("configuration", {}).get("model", PAIRED_MODEL) == model
                           and r.get("scenario", {}).get("id") == scenario_id)
        except (OSError, ValueError):
            continue
    if not matches:
        raise HTTPException(status_code=404, detail="No saved comparison for this scenario with the current protocol yet. Run a paired comparison first.")
    return max(matches, key=lambda r: r["created_at"])


@app.get('/api/compare/template/{scenario_id}')
def comparison_template(scenario_id: str):
    scenario = next((s for s in demo_scenarios() if s['id'] == scenario_id), None)
    if scenario is None:
        raise HTTPException(status_code=404, detail='Unknown template')
    return {key: scenario[key] for key in CUSTOM_FIELDS}

class RunRequest(BaseModel):
    scenario_id: str
    use_compression: bool
    compression_mode: Literal["auto", "always", "never"] = "auto"
    custom_prompt: str | None = None

@app.get("/")
def read_index():
    return FileResponse(ROOT / "index.html")

@app.get("/api/scenarios")
def get_scenarios():
    demos = [{"id": s["id"], "title": s["title"], "domain": "Paired compressor demo",
              "kind": "paired", "model": PAIRED_MODEL, "prompt": s["brief"]} for s in reversed(demo_scenarios())]
    if (ROOT / "dataset.json").exists():
        with open(ROOT / "dataset.json", "r", encoding="utf-8") as f:
            demos.extend(json.load(f))
    return demos

@app.post("/api/run")
def run_scenario(request: RunRequest):
    if request.scenario_id == "custom":
        if not request.custom_prompt or not request.custom_prompt.strip():
            raise HTTPException(status_code=422, detail="Custom prompt missing")
        task = request.custom_prompt.strip()
    else:
        # Find the scenario
        with open(ROOT / "dataset.json", "r", encoding="utf-8") as f:
            scenarios = json.load(f)
        
        scenario = next((s for s in scenarios if s["id"] == request.scenario_id), None)
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
            
        task = scenario["prompt"]
    graph = build_multi_agent_graph(use_compression=request.use_compression, compression_mode=request.compression_mode)
    
    state = initial_state(task)
    
    start_time = time.perf_counter()
    
    # Run the graph
    # To provide intermediate updates, we would stream. But for simplicity, we invoke and return the final state.
    # The frontend will display the final reports of each agent.
    try:
        final_state = graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Workflow failed: {e}") from e
        
    execution_time = time.perf_counter() - start_time
    
    return {
        "execution_time_seconds": execution_time,
        "token_usage": final_state.get("token_usage", 0),
        "compression": final_state.get("compression"),
        "report_validation": {
            role: validate_report(role, final_state[field])
            for role, field in {
                "Ingestion": "ingestion_report", "Logistics": "logistics_report",
                "Finance": "finance_report", "Risk": "risk_report",
                "Compliance": "compliance_report", "Decision": "decision_report",
                **({"Compressor": "compressed_context"} if final_state.get("compressed_context") else {}),
            }.items()
        },
        "reports": {
            "Ingestion": final_state.get("ingestion_report", ""),
            "Logistics": final_state.get("logistics_report", ""),
            "Finance": final_state.get("finance_report", ""),
            "Risk": final_state.get("risk_report", ""),
            "Compliance": final_state.get("compliance_report", ""),
            "Compressor": final_state.get("compressed_context", ""),
            "Decision": final_state.get("decision_report", "")
        }
    }

# Mount static files (css, js)
app.mount("/", StaticFiles(directory=ROOT), name="static")

