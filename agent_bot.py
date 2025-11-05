from typing import TypedDict, List
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "default"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"

class AgentState(TypedDict):
    messages: List[HumanMessage]
    response: str


llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")


def process(state: AgentState) -> AgentState:
    res = llm.invoke(state["messages"])
    print(res.content)
    state["response"] = res.content
    return state

graph = StateGraph(AgentState)
graph.add_node("processor", process)
graph.add_edge(START, "processor")
graph.add_edge("processor", END)

agent = graph.compile()

user_input = input("Enter your message: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")

