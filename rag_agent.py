from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from operator import add as add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "default"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

# embedding has to be comaptible with llm
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

file_path = "ftheta.pdf"
if not os.path.exists(file_path):
    raise Exception(f"File {file_path} does not exist.")

pdf_loader = PyPDFLoader(file_path)
# file check
try:
    pages = pdf_loader.load()
    print(f"File {file_path} contains {len(pages)} pages.")
except:
    raise Exception(f"File {file_path} is not a valid PDF file.")

# chunk text into token chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(pages)
# persist
persist_directory = "documents"
collection_name = "ftheta"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# chroma vector db

try:
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
except Exception as e:
    print(f"Error creating vector store: {str(e)}")
    raise

# retriver retrieves most similar chuncks
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # numebr of returned chunks
)


# Tool Node preparation
@tool
def retriever_tool(query: str) -> str:
    """Function tool to search and return info from document, passes to agent result """

    docs = retriever.invoke(query)
    if not docs:
        return "No info found."

    result = []
    for i, doc in enumerate(docs):
        result.append(f"Document {i + 1}: \n{doc.page_content}")
    return "\n\n".join(result)


tools = [retriever_tool]
llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# create dict for tool
tool_dict = {
    t.name: t for t in tools
}


def llm_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
                                  f"""You are a helpful pdf assistant. Use defined tools to handle pdf files.
                                        Cite specific parts of the document.                                  
                                  """
                                  )

    messages = list(state["messages"])
    messages = [system_prompt] + messages
    message = llm.invoke(messages)

    return {"messages": [message]}


def should_continue(state: AgentState) -> bool:
    """Function decides if we should continue if last message has tool call"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return True
    else:
        return False


def take_action(state: AgentState) -> AgentState:
    """Function calls retriever tool from llm response"""
    tool_calls = state["messages"][-1].tool_calls
    result = []

    for tool in tool_calls:
        print(f"Calling tool {tool['name']} with query {tool['args'].get('query', 'No queries provided')}")

        if not tool['name'] in tool_dict:
            print(f"Tool {tool['name']} not found.")
            result = "Incorrect tool call. Retry and select from available tools."
        else:
            result = tool_dict[tool['name']].invoke(tool['args'].get('query', ''))
            print(f"Tool result length: {len(str(result))}")
        #         append tool message
        result.append(ToolMessage(name=tool['name'], content=str(result), tool_call_id=tool['id']))

    print("tool complete, returning to model")
    return {"messages": [result]}


graph = StateGraph(AgentState)

graph.add_node("agent", llm_call)
graph.add_node("retriever_action", take_action)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("agent")
graph.add_edge("retriever_action", "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        True: "retriever_action",
        False: END
    }
)

rag_agent = graph.compile()


def run_rag_agent():
    """Function runs agent"""
    print("Starting agent...")

    while True:
        user_input = input("\nWhats your question: ")
        if user_input.lower() in ["quit","exit"]:
            break
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("Agent response: \n")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    run_rag_agent()
