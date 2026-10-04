import time
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# --- SETUP ---
# Use llama3, and we will extract token counts from response_metadata
llm = ChatOllama(model="llama3")

class State(TypedDict):
    messages: list[BaseMessage]
    context: str
    token_usage: int

def count_tokens(response) -> int:
    """Helper to extract token counts from Ollama response."""
    metadata = response.response_metadata
    prompt_tokens = metadata.get("prompt_eval_count", 0)
    completion_tokens = metadata.get("eval_count", 0)
    return prompt_tokens + completion_tokens

# --- AGENT DEFINITIONS ---
def researcher_node(state: State) -> dict:
    messages = state.get("messages", [])
    system_prompt = SystemMessage(content="You are a detailed researcher. Provide a comprehensive, highly detailed, and long explanation of the topic requested by the user. Do not summarize.")
    
    response = llm.invoke([system_prompt] + messages)
    tokens_used = count_tokens(response)
    
    return {
        "messages": messages + [response],
        "context": response.content,
        "token_usage": tokens_used
    }

def compressor_node(state: State) -> dict:
    context = state.get("context", "")
    
    compression_prompt = HumanMessage(
        content=f"Summarize the following text into a highly dense string, extracting only the core facts. Omit any fluff or verbose formatting.\n\nText:\n{context}"
    )
    
    response = llm.invoke([compression_prompt])
    tokens_used = count_tokens(response)
    
    return {
        "context": response.content,
        "token_usage": tokens_used
    }

def writer_node(state: State) -> dict:
    context = state.get("context", "")
    
    system_prompt = SystemMessage(content="You are a technical writer. Write a concise, 1-paragraph summary based ONLY on the provided context.")
    user_prompt = HumanMessage(content=f"Context:\n{context}\n\nPlease write your summary.")
    
    response = llm.invoke([system_prompt, user_prompt])
    tokens_used = count_tokens(response)
    
    return {
        "messages": [response],
        "token_usage": tokens_used
    }

def reviewer_node(state: State) -> dict:
    context = state.get("context", "")
    system_prompt = SystemMessage(content="You are a reviewer. Evaluate the provided context in 1 paragraph.")
    user_prompt = HumanMessage(content=f"Context:\n{context}\n\nPlease review.")
    
    response = llm.invoke([system_prompt, user_prompt])
    return {"token_usage": count_tokens(response)}

def fact_checker_node(state: State) -> dict:
    context = state.get("context", "")
    system_prompt = SystemMessage(content="You are a fact checker. Verify the provided context in 1 paragraph.")
    user_prompt = HumanMessage(content=f"Context:\n{context}\n\nPlease verify.")
    
    response = llm.invoke([system_prompt, user_prompt])
    return {"token_usage": count_tokens(response)}


# --- PIPELINE BUILDER ---
def build_pipeline(use_compression: bool):
    workflow = StateGraph(State)
    
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Reviewer", reviewer_node)
    workflow.add_node("FactChecker", fact_checker_node)
    
    if use_compression:
        workflow.add_node("Compressor", compressor_node)
        workflow.add_edge(START, "Researcher")
        workflow.add_edge("Researcher", "Compressor")
        workflow.add_edge("Compressor", "Reviewer")
        workflow.add_edge("Reviewer", "FactChecker")
        workflow.add_edge("FactChecker", "Writer")
    else:
        workflow.add_edge(START, "Researcher")
        workflow.add_edge("Researcher", "Reviewer")
        workflow.add_edge("Reviewer", "FactChecker")
        workflow.add_edge("FactChecker", "Writer")
        
    workflow.add_edge("Writer", END)
    return workflow.compile()


# --- EXPERIMENT RUNNER ---
def run_experiment():
    topic = "Explain the history, key principles, and future implications of Artificial General Intelligence (AGI)."
    
    print("==================================================")
    print("RUNNING TOKEN-EFFICIENT MULTI-AGENT EXPERIMENT")
    print("==================================================\n")
    print(f"Topic: {topic}\n")

    # 1. Baseline Pipeline (No Compression)
    print("--- 1. Baseline Pipeline (No Compression) ---")
    app_baseline = build_pipeline(use_compression=False)
    
    # We use a custom runner to track tokens step-by-step
    state_baseline = {"messages": [HumanMessage(content=topic)], "context": "", "token_usage": 0}
    
    total_tokens_baseline = 0
    start_time = time.time()
    for step in app_baseline.stream(state_baseline):
        for node_name, state_update in step.items():
            tokens = state_update.get('token_usage', 0)
            total_tokens_baseline += tokens
            print(f"[{node_name}] executed. Tokens consumed: {tokens}")
            if node_name == "Writer":
                final_output_baseline = state_update.get('messages', [])[-1].content
    
    time_baseline = time.time() - start_time
    print(f"-> Total Tokens (Baseline): {total_tokens_baseline}")
    print(f"-> Time Taken: {time_baseline:.2f}s\n")


    # 2. Token-Efficient Pipeline (With Compression)
    print("--- 2. Token-Efficient Pipeline (With Compression) ---")
    app_efficient = build_pipeline(use_compression=True)
    
    state_efficient = {"messages": [HumanMessage(content=topic)], "context": "", "token_usage": 0}
    
    total_tokens_efficient = 0
    start_time = time.time()
    for step in app_efficient.stream(state_efficient):
        for node_name, state_update in step.items():
            tokens = state_update.get('token_usage', 0)
            total_tokens_efficient += tokens
            print(f"[{node_name}] executed. Tokens consumed: {tokens}")
            if node_name == "Writer":
                final_output_efficient = state_update.get('messages', [])[-1].content

    time_efficient = time.time() - start_time
    print(f"-> Total Tokens (Efficient): {total_tokens_efficient}")
    print(f"-> Time Taken: {time_efficient:.2f}s\n")


    # 3. Results Comparison
    print("==================================================")
    print("EXPERIMENT RESULTS")
    print("==================================================")
    
    token_diff = total_tokens_baseline - total_tokens_efficient
    token_saved_pct = (token_diff / total_tokens_baseline) * 100 if total_tokens_baseline > 0 else 0
    
    print(f"Baseline Total Tokens: {total_tokens_baseline}")
    print(f"Efficient Total Tokens: {total_tokens_efficient}")
    print(f"Tokens Saved: {token_diff} ({token_saved_pct:.2f}% reduction)\n")
    
    print("--- QUALITY COMPARISON ---")
    print("BASELINE OUTPUT:")
    print(final_output_baseline.strip() + "\n")
    
    print("EFFICIENT OUTPUT:")
    print(final_output_efficient.strip() + "\n")

    print("==================================================")
    print("CONCLUSION:")
    if token_diff > 0:
        print("The compression mechanism SUCCESSFULLY reduced total token consumption.")
    else:
        print("The compression mechanism DID NOT reduce token consumption in this run.")

if __name__ == "__main__":
    run_experiment()
