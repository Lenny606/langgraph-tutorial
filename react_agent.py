from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # reducer fucntion to handle state
from langgraph.prebuilt import ToolNode

load_dotenv()
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "default"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Tool Node preparation
@tool
def add(a: int, b: int) -> int:
    """Function tool to add two numbers"""
    return a + b
@tool
def subtract(a: int, b: int) -> int:
    """Function tool to subtract two numbers"""
    return a + b
@tool
def multiply(a: int, b: int) -> int:
    """Function tool to multiply two numbers"""
    return a + b


toolList = [add, subtract, multiply]

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini").bind_tools(toolList)
# binds tools to the LLM


def llm_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are a helpful assistant. Use defined tools to answer user questions.")

    res = llm.invoke([system_prompt] + state["messages"])
    state["messages"] = [res]
    return state


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    # every message has a tool_calls attribute
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("agent", llm_call)

# create Tool Node
tool_node = ToolNode(tools=toolList)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")

agent = graph.compile()

def print_stream(stream):
    for chunk in stream:
        message = chunk["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
input = {
    "messages": [
        (
            'user', " ... add 20 and 30 . then 55 - 63 , last 5 * 8"
        )
    ]
}

print_stream(agent.stream(input, stream_mode="values"))