from typing import TypedDict, Dict
from langgraph.graph import StateGraph


# schema with typeed dict as class
class State(TypedDict):
    name: str
    days: int
    output: str

def compliment_node(state: State) -> State:
    """Simple node that compliments"""

    state["output"] = f"Hello, {state['name']}, you are doing great on your day {state["days"]}!"

    return state

agent = StateGraph(State)
agent.add_node("compliment", compliment_node)
agent.set_entry_point("compliment")
agent.set_finish_point("compliment")

app = agent.compile()
r = app.invoke({"name": "Tom", "days": 5})
print(r["output"])