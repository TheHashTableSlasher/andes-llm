import asyncio

from langchain.messages import HumanMessage, SystemMessage

__all__ = ["summary"]

async def summary_printer(stream):
    async for message in stream:
        if message["event"] == 'on_chat_model_stream':
            print(message["data"]["chunk"].content, end="", flush=True)
        elif message["event"] == 'on_chat_model_end':
            print()
            return message["data"]["output"]

def summary(model):
    system_message = """
<system_prompt>
    <role>
        Respond to the user's message, reporting on any information you retrieved and updates you made.
    </role>
    <constraints>
        <constraint>Output human-readable information -- do not hallucinate any XML or structured data.</constraint>
        <constraint>Use the chat history only to provide information on the actions you made.</constraint>
    </constraints>
</system_prompt>
"""

    def closure(state):
        messages = [SystemMessage(content=system_message)]
    
        end = -1
        
        while not isinstance(state["messages"][end], HumanMessage):
            end -= 1
    
        messages.extend(state["messages"][end:])
        
        message = asyncio.run(summary_printer(model.astream_events(messages)))
        
        return {"messages": [message]}
    
    return closure
