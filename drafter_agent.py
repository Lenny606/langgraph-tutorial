from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage, HumanMessage
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

# ,,global,, variable save into file
document_content = ""


# state into Tools as Injected State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Tool Node preparation
@tool
def update(content: str) -> str:
    """Function tool to update document with provided content"""
    global document_content
    document_content += content
    return f"Document updated with content: {content}"


@tool
def save(file_name: str) -> str:
    """Function tool to save document to file_name and finish process.

     Args:
         file_name (str): file_name name to save document to

     """

    global document_content

    if not file_name.endswith(".txt"):
        file_name = f"{file_name}.txt"

    try:
        with open(file_name, "w") as f:
            f.write(document_content)
        return f"Document saved to {file_name}"
    except Exception as e:
        return f"Error saving document to {file_name}: {str(e)}"


toolList = [save, update]

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini").bind_tools(toolList)


# binds tools to the LLM


def llm_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
                                  f"""You are a helpful writing assistant. Use defined tools to update and modify files.
                                  - if users want to save or finish document, use save tool
                                  - if users want to update document, use update tool
                                  - display current document content after modifications
                                  
                                  current document content: {document_content}
                                  """
                                  )
    # handle 1st step
    if not state["messages"]:
        user_input = "How can I help you?"
        user_message = HumanMessage(content=user_input)
        # state["messages"] = [user_message]
    else:
        user_input = input("\n What would you like to do with document")
        print(f"\n You said: {user_input}")
        user_message = HumanMessage(content=user_input)

    # combine messages - system + state + update message
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    res = llm.invoke(all_messages)

    print(f"\n Agent said: {res.content}")
    if hasattr(res, "tool_calls") and res.tool_calls:
        print(f"Using tool: {[t['name'] for t in res.tool_calls]}")

    #     return updated state (as dict)
    return {"messages": list(state["messages"]) + [user_message, res]}


def should_continue(state: AgentState) -> str:
    """Function decides if we should continue"""
    messages = state["messages"]
    if not messages:
        return "continue"

    #get most recent messages (as reversed arr)
    for message in reversed(messages):
        # if ToolMessage and contains "saved" and "document" from saved
        if isinstance(message, ToolMessage) and "save" in message.content.lower() and "document" in message.content.lower():
            return "end"
    return "continue"

def print_messages(messages: Sequence[BaseMessage]) -> None:
    """Function prints messages in pretty format"""
    if not messages:
        return

    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"Tool: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", llm_call)
graph.add_node("tools", ToolNode(tools=toolList))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)


agent = graph.compile()

def run_document_agent():
    """Function runs document agent"""
    print("Starting document agent...")

    state = { "messages": [] }

    for step in agent.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("Document agent finished.")

if __name__ == "__main__":
    run_document_agent()
