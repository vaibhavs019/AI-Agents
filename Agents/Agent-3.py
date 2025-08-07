# ReAct Agent: Reasoning and Acting agent capable of interacting with tools for performing mathematical operations (add, subtract, multiply)

from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain.tools import tool

load_dotenv("/Users/skull/PycharmProjects/AI Agents/.env")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    if a<b:
        return b - a
    else:
        return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [add, subtract, multiply]

model = init_chat_model(model="gemini-2.5-flash",
                      model_provider="google_genai").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    message = state["messages"]
    last_message = message[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {"continue": "tools",
     "end": END}
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

user_input = str(input("Enter user input: "))
inputs = {"messages": [("user", user_input)]}
while user_input != "exit":
    print_stream(app.stream(inputs, stream_mode="values"))
    user_input = str(input("Enter user input: "))
    inputs = {"messages": [("user", user_input)]}
