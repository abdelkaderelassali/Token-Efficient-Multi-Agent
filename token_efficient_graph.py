import operator
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: list[BaseMessage]
    compressed_context: str

llm = ChatOllama(model="llama3")

def Agent_A(state: State):
    messages = state.get("messages", [])
    system_prompt = SystemMessage(content="You are a detailed researcher. Provide a long, comprehensive explanation.")
    response = llm.invoke([system_prompt] + messages)
    return {"messages": messages + [response]}

def compression_node(state: State):
    messages = state.get("messages", [])
    conversation_text = "\n".join([f"{m.type}: {m.content}" for m in messages])
    
    # Calculate original context size
    original_size = len(conversation_text)

    compression_prompt = HumanMessage(
        content=f"Summarize the following conversation into a highly dense string, extracting only core facts. "
                f"Omit any fluff, pleasantries, or verbose formatting.\n\nConversation:\n{conversation_text}"
    )
    compressed_response = llm.invoke([compression_prompt])
    
    # Calculate compressed size and reduction percentage
    new_size = len(compressed_response.content)
    saved_percent = ((original_size - new_size) / original_size) * 100 if original_size > 0 else 0
    
    # Output execution metrics
    print(f"\n[METRICS] Original Context Size: {original_size} chars")
    print(f"[METRICS] Compressed Context Size: {new_size} chars")
    print(f"[METRICS] Token/Space Reduction: {saved_percent:.2f}%\n")

    return {
        "messages": [], 
        "compressed_context": compressed_response.content
    }

def Agent_B(state: State):
    compressed_context = state.get("compressed_context", "")
    system_prompt = SystemMessage(content="You are an analyst. Base your analysis solely on the provided compressed context.")
    user_prompt = HumanMessage(content=f"Context: {compressed_context}\n\nPlease provide your final analysis.")
    response = llm.invoke([system_prompt, user_prompt])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("Agent_A", Agent_A)
workflow.add_node("compression_node", compression_node)
workflow.add_node("Agent_B", Agent_B)

workflow.add_edge(START, "Agent_A")
workflow.add_edge("Agent_A", "compression_node")
workflow.add_edge("compression_node", "Agent_B")
workflow.add_edge("Agent_B", END)

app = workflow.compile()

if __name__ == "__main__":
    print("--- Initializing Token-Efficient Pipeline ---")
    print("1. Agent A is generating a verbose response...")
    initial_state = {
        "messages": [HumanMessage(content="Explain the entire history of classical mechanics and its major figures.")],
        "compressed_context": ""
    }
    final_state = app.invoke(initial_state)
    
    print("\n--- Pipeline Completed ---")
    print(f"Tokens saved by compression! Compressed Context size: {len(final_state.get('compressed_context', ''))} characters.\n")
    print("--- Final Analysis from Agent B ---")
    print("Based solely on the provided compressed context, my analysis is as follows:\n")
    print(final_state["messages"][0].content)
