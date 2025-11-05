from typing import TypedDict, List, Dict
import random
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# schema with typeed dict as class
class AgentState(TypedDict):
    name: str
    attempts: int
    guesses: List[int]
    lower_bound: int
    upper_bound: int


def greeting_node(state: AgentState) -> AgentState:
    """Simple node that compliments"""

    state["name"] = f"Hello, {state['name']} lets generate a random number!"
    state["counter"] = 0  # initialize counter
    return state


def random_node(state: AgentState) -> AgentState:
    """Simple node that generates a random number"""
    state['numberList'].append(random.randint(0, 100))
    state['counter'] += 1
    return state


def should_continue(state: AgentState) -> str:
    """Function decides if we should continue"""
    if state['counter'] < 5:
        return "loop" #edge name
    else:
        return "exit"



agent = StateGraph(AgentState)
agent.add_node("greeter", greeting_node)
agent.add_node("random", random_node)

agent.add_edge("greeter", "random")
agent.add_conditional_edges(
    "random",
    should_continue,
    {
        "loop": "random",
        "exit": END  #dont need define exit_point
    }
)

agent.set_entry_point("greeter")

app = agent.compile()

# display(Image(app.get_graph().draw_mermaid_png(output_file_path="./images/looping_graph.png")))
r = app.invoke({"name": "Tom", "numberList": [], "counter": 0})
print(r["numberList"])
