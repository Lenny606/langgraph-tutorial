import random
from typing import TypedDict,List
from langgraph.graph import StateGraph
from IPython.display import Image, display


class AgentState(TypedDict):
    name: str
    age: str
    skills: List[str]
    final: str


def first_node(state: AgentState) -> AgentState:
    """First Node, Simple node that adds greeting"""

    state["final"] = f"hello {state['name']}, "
    return state


def second_node(state: AgentState) -> AgentState:
    """Second Node, Simple node that adds age"""

    state["final"] = state['final'] + f"your age is:  {state['age']}, "
    return state

def third_node(state: AgentState) -> AgentState:
    """Second Node, Simple node that adds skill"""

    random_int = random.randint(0, 4)

    state["final"] = state['final'] + f"your skill is:  {state['skills'][random_int]}"
    return state


graph = StateGraph(AgentState)
graph.add_node("first", first_node)
graph.add_node("second", second_node)
graph.add_node("third", third_node)

graph.set_entry_point("first")
# add edges
graph.add_edge("first", "second")
graph.add_edge("second", "third")
graph.set_finish_point("third")

app = graph.compile()

display(Image(app.get_graph().draw_mermaid_png(output_file_path="./images/sequential.png")))

list_of_skills = ["python", "java", "c++", "c#", "javascript"]

result = app.invoke({"name": "Tom", "age": 33, "skills": list_of_skills})
print(result['final'])