from typing import TypedDict, Dict
from langgraph.graph import StateGraph
from IPython.display import Image, display


# schema with typeed dict as class
class AgentState(TypedDict):
    message: str


# node as function, doctypes important for llm
def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds greeting"""

    state["message"] = f"Hello, {state['message']}, whats up!"

    return state


# build graph
graph = StateGraph(AgentState)
graph.add_node("greeter", greeting_node)
graph.set_entry_point("greeter")
graph.set_finish_point("greeter")
app = graph.compile()

# visualisation
display(Image(app.get_graph().draw_mermaid_png(output_file_path="./images/graph.png")))

result = app.invoke({"message": "Tom"})
print(result["message"])
