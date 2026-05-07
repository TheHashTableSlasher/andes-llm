import andes
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.graph import MessagesState
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain_ollama import ChatOllama
from pprint import pp
import sys

class MyState(MessagesState):
    case: andes.system.System

@deprecated
@tool
def load_case(case: str, runtime: ToolRuntime) -> Command:
    """Load a test case for a power system simulation"""
    ss = andes.load(andes.get_case('kundur/kundur_full.xlsx'))
    return Command(
        update={
            "case": ss,
            "messages": [
                ToolMessage(
                    content="Success",
                    tool_call_id=runtime.tool_call_id # Access via runtime
                    )
            ]
        }
    )

@tool
def get_model(prop: str, runtime: ToolRuntime) -> str:
    """Get a specific model in an ANDES system. A system must be loaded first."""
    ss = runtime.state["case"]
    return ss.models[prop].as_df().to_markdown()
    
@tool
def get_pflow(runtime: ToolRuntime) -> None:
    pass
    
@tool
def get_tds(runtime: ToolRuntime) -> None:
    pass
    
@tool
def get_eig(runtime: ToolRuntime) -> None:
    pass

@deprecated
@tool
def run_simulation(runtime: ToolRuntime):
    """Run the power flow simulation for the loaded power system simulation."""
    ss = runtime.state["case"]
    print("Running simulation...", file=sys.stderr)
    ss.PFlow.run()
