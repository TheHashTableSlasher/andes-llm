import andes
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def load_test_case(path: str, runtime: ToolRuntime) -> Command:
    """Load a preconfigured test case for an ANDES power system."""
    update = {}
    
    try:
        ss = andes.load(andes.get_case(path))
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
        ss = andes.load(path)
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
    
    model = model.bind_tools([load_local_case, load_test_case])

    def closure(state):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["steps"][0])
        ]
        
        messages.append(model.invoke(messages))
        
        if state.get("debug", False):
            print(f"\033[31mload_system: will attempt the following tool calls: \"{messages[-1].tool_calls}\"\033[0m")
        
        return {"messages": messages}
    
    return closure
