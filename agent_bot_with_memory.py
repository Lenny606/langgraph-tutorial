from typing import TypedDict, List, Union
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "default"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    response: str


llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")


def process(state: AgentState) -> AgentState:
    """Function node that processes user input"""
    res = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=res.content))
    print(res.content)
    return state


graph = StateGraph(AgentState)
graph.add_node("processor", process)
graph.add_edge(START, "processor")
graph.add_edge("processor", END)

agent = graph.compile()

# HISTORY
conversation_history = []

user_input = input("Enter your message: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    response = agent.invoke({"messages": conversation_history})
    conversation_history = response["messages"]

    user_input = input("Enter your message: ")

with open("conversation_history.txt", "w") as f:
    f.write("Starting conversation history:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"Human: {message.content}\n")
        else:
            f.write(f"AI: {message.content}\n")

    f.write("End of conversation.\n")
    f.close()