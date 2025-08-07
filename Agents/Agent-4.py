# Drafter agent: Draft a response to a user's request

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv("/Users/skull/PycharmProjects/AI Agents/.env")

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update document content"""
    global document_content
    document_content = content
    return f"Document content updated successfully, new content is: \n{document_content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"

tools = [update, save]

model = init_chat_model(model="gemini-2.5-flash",
                      model_provider="google_genai").bind_tools(tools)

def our_agent(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state['messages']:
        user_input = "I'm ready to help you update a document."
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("What would you like to do with this document? ")
        print(f"User input: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]

    response = model.invoke(all_messages)
    print(f"AI response: {response.content}")
    if hasattr(response, "tool_calls"):
        print("Tool calls detected:", response.tool_calls)

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """This functions will determine if the agent should continue or not"""
    messages = state["messages"]

    if not messages:
        return "continue"

    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
        "saved" in message.content.lower() and
        "document" in message.content.lower()):
            return "end"

    return "continue"

def print_messages(messages):
    """This function will print the messages"""
    if not messages:
        return

    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"Tool result: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools=tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {"continue": "agent",
     "end": END}
)

app = graph.compile()

def main():
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        print_messages(step["messages"])
    print("\n ==== FINAL STATE ====")

if __name__ == "__main__":
    main()