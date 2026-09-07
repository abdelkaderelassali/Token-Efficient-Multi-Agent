import logging
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: list[BaseMessage]
    compressed_context: str

llm = ChatOllama(model="llama3")

def data_ingestion_node(state: State) -> dict:
    messages = state.get("messages", [])
    system_prompt = SystemMessage(content="You are a data ingestion specialist. Extract and formalize key raw data points from the request.")
    try:
        response = llm.invoke([system_prompt] + messages)
        return {"messages": messages + [response]}
    except Exception as e:
        return {"messages": messages + [SystemMessage(content="Data ingestion failed.")]}

def logistics_planner_node(state: State) -> dict:
    messages = state.get("messages", [])
    system_prompt = SystemMessage(content="You are a logistics planner. Construct a detailed logistical timeline based on prior data.")
    try:
        response = llm.invoke([system_prompt] + messages)
        return {"messages": messages + [response]}
    except Exception as e:
        return {"messages": messages + [SystemMessage(content="Logistics planning failed.")]}

def context_compressor_node(state: State) -> dict:
    messages = state.get("messages", [])
    conversation_text = "\n".join([f"{m.type}: {m.content}" for m in messages])
    compression_prompt = HumanMessage(content=f"Summarize the core facts of the following conversation into a dense block of text.\n\nConversation:\n{conversation_text}")
    try:
        compressed_response = llm.invoke([compression_prompt])
        compressed_text = compressed_response.content
    except Exception as e:
        compressed_text = "Compression failed."
    return {"messages": [], "compressed_context": compressed_text}

def financial_analyst_node(state: State) -> dict:
    compressed_context = state.get("compressed_context", "")
    system_prompt = SystemMessage(content="You are a financial analyst. Base your fiscal analysis strictly on the provided compressed context.")
    user_prompt = HumanMessage(content=f"Context: {compressed_context}\n\nProvide financial estimates.")
    try:
        response = llm.invoke([system_prompt, user_prompt])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [SystemMessage(content="Financial analysis failed.")]}

def decision_manager_node(state: State) -> dict:
    messages = state.get("messages", [])
    system_prompt = SystemMessage(content="You are the decision manager. Provide a final executive ruling based on the financial analysis.")
    try:
        response = llm.invoke([system_prompt] + messages)
        return {"messages": messages + [response]}
    except Exception as e:
        return {"messages": messages + [SystemMessage(content="Decision management failed.")]}

workflow = StateGraph(State)
workflow.add_node("DataIngestionNode", data_ingestion_node)
workflow.add_node("LogisticsPlannerNode", logistics_planner_node)
workflow.add_node("ContextCompressorNode", context_compressor_node)
workflow.add_node("FinancialAnalystNode", financial_analyst_node)
workflow.add_node("DecisionManagerNode", decision_manager_node)

workflow.add_edge(START, "DataIngestionNode")
workflow.add_edge("DataIngestionNode", "LogisticsPlannerNode")
workflow.add_edge("LogisticsPlannerNode", "ContextCompressorNode")
workflow.add_edge("ContextCompressorNode", "FinancialAnalystNode")
workflow.add_edge("FinancialAnalystNode", "DecisionManagerNode")
workflow.add_edge("DecisionManagerNode", END)

app = workflow.compile()

if __name__ == "__main__":
    initial_state: State = {
        "messages": [HumanMessage(content="Plan a new supply chain route from New York to London including shipping volume estimates.")],
        "compressed_context": ""
    }
    final_state = app.invoke(initial_state)
    final_message = final_state.get("messages", [])[-1].content if final_state.get("messages") else ""
    print(final_message)
