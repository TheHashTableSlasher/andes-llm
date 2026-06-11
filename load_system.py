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
    system_message = """
<system_prompt>
    <role>
        Use the <tool>load_test_case</tool> and <tool>load_local_case</tool> tools to load the ANDES system that the user has specified.
    </role>
    <constraints>
        <constraint>Only make a single tool call.</constraint>
        <constraint>Do not provide any additional output, aside from the tool call.</constraint>
        <constraint>If a user includes a path in the prompt, pass that path verbatim. Do not corrupt that path in any way.</constraint>
    </constraints>
</system_prompt>
"""
    
    model = model.bind_tools([load_local_case, load_test_case])

    def closure(state):
        messages = [SystemMessage(content=system_message)]
        end = -1
        
        while not isinstance(state["messages"][end], HumanMessage):
            end -= 1
    
        messages.extend(state["messages"][(end + 1):])
        messages.append(HumanMessage(content=state["steps"][0]))
        
        message = model.invoke(messages)
        
        if state.get("debug", False):
            print(f"\033[31mload_system: will attempt the following tool calls: \"{message.tool_calls}\"\033[0m")
        
        return {"messages": [message]}
    
    return closure
