import andes
import json

from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

__all__ = ["get_model", "question_param"]

with open("question_param.json", "r") as file:
    param_info = json.load(file)

@tool
def get_model(modelname: str, state: Annotated[dict, InjectedState]) -> str:
    """Look up information about a specific model."""
    
    param_data = state["ss"].models.get(modelname)
    
    if param_data is None:
        return ""
    
    param_info_str = param_info.get(modelname, "")
    param_data_str = param_data.as_df().to_markdown()
    
    return param_info_str + "Current values\n--------------\n\n\n" + param_data_str

def question_param(model):
    system_message = """
<system_prompt>
    <role>
        Use the <tool>get_model</tool> tool to look up information about a specific ANDES model. You may call this tool as many times as you like.
    </role>
</system_prompt>
"""
    
    model = model.bind_tools([get_model])

    def closure(state):
        messages = [SystemMessage(content=system_message)]
        end = -1
        
        while not isinstance(state["messages"][end], HumanMessage):
            end -= 1
    
        messages.extend(state["messages"][(end + 1):])
        messages.append(HumanMessage(content=state["steps"][0]))
        
        message = model.invoke(messages)
        
        if state.get("debug", False):
            print(f"\033[31mquestion_param: will attempt the following tool calls: \"{message.tool_calls}\"\033[0m")
        
        return {"messages": [message]}
    
    return closure
