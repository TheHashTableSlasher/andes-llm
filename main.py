import operator
import os
import readline

from andes.system import System
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated

from .classifier import classifier

class State(MessagesState):
    messages: Annotated[list, add_messages]
    steps: List[str]
    next: str
    ss: System
    
def planner(state):
    # TODO: use this step to split into compounds, conditionals, etc.
    state["steps"] = [state["messages"][-1].content]

if __name__ == "__main__":
    model = ChatOllama(
        model="llama3.1:8b",
        temperature=0,
    )
    
    graph = StateGraph(State)
    tools = []
    
    graph.add_node("planner", lambda state: state)
    graph.add_node("next_step", lambda state: state)
    graph.add_node("classifier", classifier(model))
    
    graph.add_node("question_general", lambda state: state) # TODO
    graph.add_node("question_param", lambda state: state) # TODO
    graph.add_node("load_system", load_system(model))
    graph.add_node("run_pflow", lambda state: state) # TODO
    graph.add_node("run_tds", lambda state: state) # TODO
    graph.add_node("run_eig", lambda state: state) # TODO
    graph.add_node("interpreter", lambda state: state) # TODO
    
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("summary", lambda state: state)
    
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "next_step")
    graph.add_conditional_edges("next_step", lambda state: len(state["steps"]) > 0, {True: "classifier", False: "summary"})
    graph.add_conditional_edges("classifier", operator.itemgetter("next"))
    
    graph.add_edge("question_general", "tools")
    graph.add_edge("question_param", "tools")
    graph.add_edge("load_system", "tools")
    graph.add_edge("run_pflow", "tools")
    graph.add_edge("run_tds", "tools")
    graph.add_edge("run_eig", "tools")
    graph.add_edge("interpreter", "tools")
    
    graph.add_edge("tools", "next_step")
    
    graph.add_edge("summary", END)
    
    graph = graph.compile()
    
    state = State()
    state["messages"] = []
    
    print("ANDES LLM v0.0.?")
    print("See LICENSE for license info")

    histfile = ".llm_history"
    if os.path.exists(histfile):
        readline.read_history_file(histfile)
        
    try:
        while True:
            try:
                content = input("> ")
                state["messages"].append(HumanMessage(content=content))
                state = graph.invoke(state)
            except KeyboardInterrupt:
                print()
                break
    finally:
        readline.write_history_file(histfile)
