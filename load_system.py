from langchain.messages import SystemMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def load_test_case(path: str, runtime: ToolRuntime) -> Command:
    """Load a preconfigured test case for an ANDES power system."""
    update = {}
    
    try:
        ss = andes.load(andes.get_case(case))
    except Exception as err:
        update["messages"] = [ToolMessage(
            status="error",
            content=err,
            tool_call_id=runtime.tool_call_id
        )]
    else:
        update["ss"] = ss
        update["messages"] = [ToolMessage(
            status="success",
            content=ss,
            tool_call_id=runtime.tool_call_id
        )]
   
    return Command(update=update)

@tool
def load_local_case(path: str, runtime: ToolRuntime) -> Command:
    """Load a real ANDES power system from a file."""
    update = {}
    
    try:
        ss = andes.load(case)
    except Exception as err:
        update["messages"] = [ToolMessage(
            status="error",
            content=err,
            tool_call_id=runtime.tool_call_id
        )]
    else:
        update["ss"] = ss
        update["messages"] = [ToolMessage(
            status="success",
            content=ss,
            tool_call_id=runtime.tool_call_id
        )]
   
    return Command(update=update)

def load_system(model):
    system_prompt = "You are a helpful power systems assistant that has access to tools for loading and retrieving the properties of a power systems test case."

    #messages = [
    #    SystemMessage(content=system_prompt),
    #    state["messages"][-1]
    #]
    
    return lambda state: state # TODO
