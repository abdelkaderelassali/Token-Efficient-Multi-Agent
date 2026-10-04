import json
import time
import operator
from report_format import report_instructions
from context_routing import route_context
from compression_policy import compression_policy
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

llm = ChatOllama(model="llama3")

def count_tokens(response) -> int:
    metadata = response.response_metadata
    for key in ("prompt_eval_count", "eval_count"):
        if type(metadata.get(key)) is not int or metadata[key] < 0:
            raise ValueError(f"Missing or invalid Ollama token measurement: {key}")
    return metadata["prompt_eval_count"] + metadata["eval_count"]

def report_text(response, role: str) -> str:
    """Reject missing reports before downstream agents can consume them."""
    text = response.content
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{role}: model returned an empty or non-text report")
    return text


# --- STATE DEFINITION ---
class State(TypedDict):
    task: str
    ingestion_report: str
    logistics_report: str
    finance_report: str
    compressed_context: str
    compression: dict | None
    risk_report: str
    compliance_report: str
    decision_report: str
    token_usage: Annotated[int, operator.add]


def initial_state(task: str) -> State:
    return {
        "task": task,
        "ingestion_report": "",
        "logistics_report": "",
        "finance_report": "",
        "compressed_context": "",
        "compression": None,
        "risk_report": "",
        "compliance_report": "",
        "decision_report": "",
        "token_usage": 0,
    }


def specialist_context(state: State) -> str:
    """Select downstream context without modifying the original reports."""
    if state.get("compressed_context"):
        return state["compressed_context"]
    return (
        f"LOGISTICS:\n{state.get('logistics_report', '')}\n\n"
        f"FINANCE:\n{state.get('finance_report', '')}"
    )

# --- SINGLE AGENT BASELINE ---
def single_agent_run(task: str) -> tuple[str, int, float]:
    start_time = time.perf_counter()
    system_prompt = SystemMessage(content="You are an AI assistant. Solve the user's logistics and finance problem completely.")
    user_prompt = HumanMessage(content=task)
    
    response = llm.invoke([system_prompt, user_prompt])
    tokens = count_tokens(response)
    
    return response.content, tokens, time.perf_counter() - start_time

# --- MULTI-AGENT NODES ---
def ingestion_node(state: State) -> dict:
    prompt = SystemMessage(content="Extract the key entities, locations, and constraints from the task.")
    prompt = SystemMessage(content=prompt.content + report_instructions("Ingestion"))
    response = llm.invoke([prompt, HumanMessage(content=state["task"])])
    return {"ingestion_report": report_text(response, "Ingestion"), "token_usage": count_tokens(response)}

def logistics_node(state: State) -> dict:
    context = f"ORIGINAL REQUEST:\n{state['task']}\n\nEXTRACTED DATA:\n{state['ingestion_report']}"
    prompt = SystemMessage(content="You are a Logistics Expert. Plan the route and shipping method using the original request as authoritative over extracted data. Do not invent missing facts.")
    prompt = SystemMessage(content=prompt.content + report_instructions("Logistics"))
    response = llm.invoke([prompt, HumanMessage(content=f"Data:\n{context}")])
    return {"logistics_report": report_text(response, "Logistics"), "token_usage": count_tokens(response)}

def finance_node(state: State) -> dict:
    context = f"ORIGINAL REQUEST:\n{state['task']}\n\nEXTRACTED DATA:\n{state['ingestion_report']}"
    prompt = SystemMessage(content="You are a Finance Expert. Calculate costs and verify budget constraints using the original request as authoritative over extracted data. Do not invent missing facts. Distinguish estimates from supplied costs.")
    prompt = SystemMessage(content=prompt.content + report_instructions("Finance"))
    response = llm.invoke([prompt, HumanMessage(content=f"Data:\n{context}")])
    return {"finance_report": report_text(response, "Finance"), "token_usage": count_tokens(response)}

def compression_node(state: State, mode="auto") -> dict:
    logistics = state.get("logistics_report", "")
    finance = state.get("finance_report", "")
    combined = f"LOGISTICS:\n{logistics}\n\nFINANCE:\n{finance}"
    
    policy = compression_policy(combined, consumers=3, mode=mode)
    if not policy["should_compress"]:
        return {"compressed_context": "", "compression": policy, "token_usage": 0}
    system_prompt = SystemMessage(content="You are an expert Context Compressor. Your goal is to drastically reduce token count while maintaining 100% of the factual information. You MUST retain all numbers, dates, locations, budgets, costs, and constraints verbatim. Output a highly dense bulleted list containing only the core facts.")
    prompt = HumanMessage(content=f"Compress these reports:\n\n{combined}")
    system_prompt = SystemMessage(content=system_prompt.content + report_instructions("Compressor"))
    response = llm.invoke([system_prompt, prompt])
    
    compact = report_text(response, "Compressor")
    policy.update(compressor_called=True, applied=mode == "always" or len(compact) < len(combined))
    if not policy["applied"]:
        policy["reason"] = "Generated context was not shorter; kept full context and counted compressor cost"
    return {"compressed_context": compact if policy["applied"] else "", "compression": policy, "token_usage": count_tokens(response)}

def decision_node(state: State) -> dict:
    context = (
        f"ORIGINAL REQUEST:\n{state['task']}\n\n"
        f"SPECIALIST FINDINGS:\n{specialist_context(state)}\n\n"
        f"RISK REVIEW:\n{state['risk_report']}\n\n"
        f"COMPLIANCE REVIEW:\n{state['compliance_report']}"
    )
    prompt = SystemMessage(content=(
        "You are the Decision Manager. Provide the final executive decision using the "
        "original request, specialist findings, and BOTH the Risk and Compliance reviews. "
        "State whether to proceed, proceed conditionally, or hold. Explain how the reviews "
        "affect your decision. Resolve conflicting recommendations where the supplied "
        "evidence allows; otherwise identify the conflict and required verification. "
        "Treat the original request as authoritative for user constraints. Distinguish "
        "estimates and assumptions from verified facts. Do not invent missing facts or "
        "claim regulatory approval without evidence. Include conditions and next actions."
    ))
    prompt = SystemMessage(content=prompt.content + report_instructions("Decision"))
    response = llm.invoke([prompt, HumanMessage(content=f"Context:\n{context}")])
    return {"decision_report": report_text(response, "Decision"), "token_usage": count_tokens(response)}

def risk_node(state: State) -> dict:
    context, _ = route_context(
        "Risk", f"ORIGINAL REQUEST:\n{state['task']}",
        {"Logistics": state["logistics_report"], "Finance": state["finance_report"]},
        f"ORIGINAL REQUEST:\n{state['task']}\n\n{state['compressed_context']}" if state["compressed_context"] else None,
    )
    prompt = SystemMessage(content="You are the Risk Assessor. Identify risks against the original request, which overrides generated findings. Include unresolved constraints in your report.")
    prompt = SystemMessage(content=prompt.content + report_instructions("Risk"))
    response = llm.invoke([prompt, HumanMessage(content=f"Context:\n{context}")])
    return {"risk_report": report_text(response, "Risk"), "token_usage": count_tokens(response)}

def compliance_node(state: State) -> dict:
    context = f"ORIGINAL REQUEST:\n{state['task']}\n\nSPECIALIST FINDINGS:\n{specialist_context(state)}"
    prompt = SystemMessage(content="You are the Compliance Officer. Check the findings against the original request and its constraints. Do not invent regulations or claim approval without evidence. Identify missing evidence.")
    prompt = SystemMessage(content=prompt.content + report_instructions("Compliance"))
    response = llm.invoke([prompt, HumanMessage(content=f"Context:\n{context}")])
    return {"compliance_report": report_text(response, "Compliance"), "token_usage": count_tokens(response)}

# --- GRAPH BUILDER ---
def build_multi_agent_graph(use_compression: bool, compression_mode="auto"):
    if compression_mode not in ("auto", "always", "never"):
        raise ValueError("compression mode must be auto, always, or never")
    workflow = StateGraph(State)
    
    workflow.add_node("Ingestion", ingestion_node)
    workflow.add_node("Logistics", logistics_node)
    workflow.add_node("Finance", finance_node)
    workflow.add_node("Decision", decision_node)
    workflow.add_node("Risk", risk_node)
    workflow.add_node("Compliance", compliance_node)
    
    workflow.add_edge(START, "Ingestion")
    
    # Parallel execution for logistics and finance
    workflow.add_edge("Ingestion", "Logistics")
    workflow.add_edge("Ingestion", "Finance")
    
    if use_compression:
        workflow.add_node("Compressor", lambda state: compression_node(state, compression_mode))
        workflow.add_edge(["Logistics", "Finance"], "Compressor")
        workflow.add_edge("Compressor", "Risk")
        workflow.add_edge("Compressor", "Compliance")
    else:
        workflow.add_edge(["Logistics", "Finance"], "Risk")
        workflow.add_edge(["Logistics", "Finance"], "Compliance")
        
    # Both independent reviews must finish before the executive decision.
    workflow.add_edge(["Risk", "Compliance"], "Decision")
    workflow.add_edge("Decision", END)
    return workflow.compile()

# --- EVALUATION ---
def check_constraints(text: str, constraints: list[str]) -> float:
    """Legacy keyword coverage only; presence does not prove factual correctness."""
    text_lower = text.lower()
    matches = sum(1 for c in constraints if c.lower() in text_lower)
    return (matches / len(constraints)) * 100 if constraints else 100.0

# --- BENCHMARK RUNNER ---
def run_benchmark():
    with open("dataset.json", "r") as f:
        scenarios = json.load(f)
        
    results = []
    
    app_uncompressed = build_multi_agent_graph(use_compression=False)
    app_compressed = build_multi_agent_graph(use_compression=True)
    
    for i, scenario in enumerate(scenarios):
        print(f"\n======================================")
        print(f"RUNNING SCENARIO {i+1}: {scenario['id']}")
        print(f"======================================")
        
        task = scenario["prompt"]
        constraints = scenario["expected_constraints"]
        
        # 1. Single Agent Baseline
        print("-> Running Baseline 1: Single Agent...")
        sa_out, sa_tokens, sa_time = single_agent_run(task)
        sa_quality = check_constraints(sa_out, constraints)
        
        # 2. Multi-Agent Uncompressed Baseline
        print("-> Running Baseline 2: Multi-Agent (Uncompressed)...")
        state_uncompressed = initial_state(task)
        start_time = time.perf_counter()
        final_uncomp = app_uncompressed.invoke(state_uncompressed)
        mu_time = time.perf_counter() - start_time
        mu_tokens = final_uncomp["token_usage"]
        mu_out = final_uncomp["decision_report"]
        mu_quality = check_constraints(mu_out, constraints)
        
        # 3. Multi-Agent Compressed (Proposed System)
        print("-> Running Proposed System: Multi-Agent (Compressed)...")
        state_compressed = initial_state(task)
        start_time = time.perf_counter()
        final_comp = app_compressed.invoke(state_compressed)
        mc_time = time.perf_counter() - start_time
        mc_tokens = final_comp["token_usage"]
        mc_out = final_comp["decision_report"]
        mc_quality = check_constraints(mc_out, constraints)
        
        # Save results
        results.append({
            "scenario": scenario["id"],
            "Single_Agent": {"tokens": sa_tokens, "time": sa_time, "constraint_mention_coverage_percent": sa_quality},
            "Multi_Agent_Uncompressed": {"tokens": mu_tokens, "time": mu_time, "constraint_mention_coverage_percent": mu_quality},
            "Multi_Agent_Compressed": {"tokens": mc_tokens, "time": mc_time, "constraint_mention_coverage_percent": mc_quality}
        })
        
        print(f"\n[Scenario {i+1} Summary]")
        print(f"Tokens: SA={sa_tokens}, MU={mu_tokens}, MC={mc_tokens}")
        print(f"Constraint mention coverage (not correctness): SA={sa_quality}%, MU={mu_quality}%, MC={mc_quality}%")
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nBenchmark completed! Results saved to results.json.")

if __name__ == "__main__":
    run_benchmark()
