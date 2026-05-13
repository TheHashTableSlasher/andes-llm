from langchain.messages import HumanMessage, SystemMessage

__all__ = ["summary"]

def summary(model):
    system_message = """
You are an assistant for ANDES, a library for power system modeling and simulation. Respond to the user's message, reporting on any information you retrieved and updates you made.
"""

    def closure(state):
        end = -1
        
        while not isinstance(state["messages"][end], HumanMessage):
            end -= 1
    
        messages = [SystemMessage(content=system_message)] + state["messages"][end:]
        
        message = model.invoke(messages)
        
        print(message.content)
        
        return {"messages": [message]}
    
    return closure
