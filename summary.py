import asyncio

from langchain.messages import HumanMessage, SystemMessage

__all__ = ["summary"]

async def summary_printer(stream):
    async for message in stream:
        if message["event"] == 'on_chat_model_stream':
            print(message["data"]["chunk"].content, end="", flush=True)
        elif message["event"] == 'on_chat_model_end':
            return message["data"]["output"]

def summary(model):
    system_message = """
You are an assistant for ANDES, a library for power system modeling and simulation. Respond to the user's message, reporting on any information you retrieved and updates you made.
"""

    def closure(state):
        end = -1
        
        while not isinstance(state["messages"][end], HumanMessage):
            end -= 1
    
        messages = state["messages"][end:]
        
        message = asyncio.run(summary_printer(model.astream_events(messages)))
        
        return {"messages": [message]}
    
    return closure
