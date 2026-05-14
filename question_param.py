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
    system_prompt = "You are a helpful power systems assistant that has access to tools for looking up information about a specific ANDES model."
    
    model = model.bind_tools([get_model])

    def closure(state):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["steps"][0])
        ]
        
        messages.append(model.invoke(messages))
        
        if state.get("debug", False):
            print(f"\033[31mquestion_param: will attempt the following tool calls: \"{messages[-1].tool_calls}\"\033[0m")
        
        return {"messages": messages}
    
    return closure
