from langchain_ollama import ChatOllama
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain.messages import HumanMessage, SystemMessage

__all__ = ["classifier"]

class MessageClassifier(BaseModel):
    category: Literal["question_general", "question_param", "load_system", "codegen"] = Field(..., description="A classification of the kind of sentence that the message is.")
    justification: str = Field(..., description="A brief rationale of your thought process that led you to the classification that you selected.")

def classifier(model):
    system_message = """
<system_prompt>
    <role>
        Categorize the user's message into one of the following categories:
        
        <category name="question_general">
            An educational query about some aspect of power systems generally. These questions may not be specific to any aspect of ANDES.
            <example>What is a PV curve?</example>
            <example>Explain to me what eigenvalue analysis does</example>
            <example>How do I interpret the results of a time-domain simulation?</example>
        </category>
        
        <category name="question_param">
            An educational query about some aspect of an ANDES system loaded elsewhere.
            <example>Tell me about the system's PFlow member</example>
            <example>Analyze the final results of the simulation</example>
            <example>What is the final result at each bus?</example>
        </category>
        
        <category name="load_system">
            A command to load a new ANDES system from some path.
            <example>Load the test case kundur/kundur_full.xlsx</example>
            <example>Load the system my_custom_setup.json from the current directory</example>
        </category>
        
        <category name="codegen">
            Instructions to perform some actions with an ANDES system. Any commands that do not belong in one of the previous categories should be placed into this category.
        </category>
    </role>
    <constraints>
        <constraint>Do not provide any additional output, aside from the categorization and justification.</constraint>
    </constraints>
</system_prompt>
"""

    model = model.with_structured_output(MessageClassifier, include_raw=True)
    
    def closure(state):
        messages = [SystemMessage(content=system_message)]
        end = -1
        
        while not isinstance(state["messages"][end], HumanMessage):
            end -= 1
    
        messages.extend(state["messages"][(end + 1):])
        messages.append(HumanMessage(content=state["steps"][0]))
        
        message = model.invoke(messages)
        
        raw = message["raw"]
        parsed = message["parsed"]
        
        messages.append(raw)
        
        if state.get("debug", False):
            print(f"\033[31mclassifier: classified as {parsed.category} with the following justification: \"{parsed.justification}\"\033[0m")
        
        return {"messages": [raw], "next": parsed.category}
    
    return closure

#print(.invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "What is a power-flow simulation?"}]))
#print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "What is the current value of the system's #eigenvalues?"}]))
#print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "Load a random test case and run a power-flow simulation"}]))
#print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "Load a random test case and run a power-flow simulation"}]))

