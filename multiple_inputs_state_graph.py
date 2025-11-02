from typing import TypedDict, List
from langgraph.graph import StateGraph
from IPython.display import Image, display

class AgentState(TypedDict):
    values: List[int]
    name: str
    result: str

def process_values(state: AgentState) -> AgentState:
    """Function handles multiple inputs"""

    state["result"] = f"Hello, {state['name']}, your sum is {sum(state['values'])}"

    return state

graph = StateGraph(AgentState)
graph.add_node("processor", process_values)
graph.set_entry_point("processor")
graph.set_finish_point("processor")
app = graph.compile()

display(Image(app.get_graph().draw_mermaid_png(output_file_path="./images/multi_input_graph.png")))

result = app.invoke({"name": "Tom", "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
print(result["result"])
