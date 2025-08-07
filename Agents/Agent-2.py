# Agent: Chat with LLM and store conversation history

from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv("/Users/skull/PycharmProjects/AI Agents/.env")

# For gemini
llm = init_chat_model(model="gemini-2.5-flash",
                      model_provider="google_genai")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def process(state:AgentState) -> AgentState:
    """"This node will solve the request your user has made"""""
    response = llm.invoke(state['messages'])

    state['messages'].append(AIMessage(content=response.content))

    print(f"Response: {response.content}")
    # print("Current state: ", state['messages'])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter user input: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result['messages']
    user_input = input("Enter user input: ")

with open("conversation_history.txt", "w") as file:
    file.write("Your conversation history:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation history.")
print("Conversation history saved to conversation_history.txt")