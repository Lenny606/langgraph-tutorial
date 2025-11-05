from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


class AgentState(TypedDict):
    num1: int
    operation: str
    operation2: str
    num2: int
    result: int


def adder(state: AgentState) -> AgentState:
    """Function adds inputs"""
    state["result"] = state['num1'] + state['num2']
    print(state["result"])
    return state


def subtractor(state: AgentState) -> AgentState:
    """Function subs inputs"""
    state["result"] = state['num1'] - state['num2']
    print(state["result"])
    return state


def decide_next_node(state: AgentState) -> str:
    """Function decides which node to run"""
    if state["operation"] == "+":
        return "addition_ops"  # returns edge
    else:
        return "substraction_ops"


def decide_next_node2(state: AgentState) -> str:
    """Function decides which node to run"""
    if state["operation2"] == "+":
        return "addition_ops2"  # returns edge
    else:
        return "substraction_ops2"


graph = StateGraph(AgentState)

graph.add_node("adder_node", adder)
graph.add_node("adder_node2", adder)
graph.add_node("substractor_node", subtractor)
graph.add_node("substractor_node2", subtractor)
graph.add_node("router", lambda state: state)
graph.add_node("router2", lambda state: state)

graph.add_edge(START, 'router')

graph.add_conditional_edges(
    "router",
    decide_next_node,
    {
        # Edge: Node format
        "addition_ops": "adder_node",
        "substraction_ops": "substractor_node"
    }
)

graph.add_edge("adder_node", "router2")
graph.add_edge("substractor_node", "router2")

graph.add_conditional_edges(
    "router2",
    decide_next_node2,
    {
        # Edge: Node format
        "addition_ops2": "adder_node2",
        "substraction_ops2": "substractor_node2"
    }

)

graph.add_edge("adder_node2", END)
graph.add_edge("substractor_node2", END)


app = graph.compile()

display(Image(app.get_graph().draw_mermaid_png(output_file_path="./images/exercise_conditional_graph.png")))

init_state = AgentState(num1=5, operation="-", operation2="+", num2=3)

result = app.invoke(init_state)
print(result["result"])
