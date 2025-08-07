# Agent: Chat with LLM

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv("/Users/skull/PycharmProjects/AI Agents/.env")

class AgentState(TypedDict):
    messages: List[HumanMessage]

# For gemini
llm = init_chat_model(model="gemini-2.5-flash",
                      model_provider="google_genai")

# ChatGPT
# llm = ChatOpenAI(model="gpt-4o")

def process(state:AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    print(f"Response: {response}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter user input: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter user input: ")