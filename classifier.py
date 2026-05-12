from langchain_ollama import ChatOllama
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from typing_extensions import Literal

class MessageClassifier(BaseModel):
    category: Literal["question_general", "question_param", "load_system", "interpreter"] = Field(..., description="A classification of the kind of sentence that the message is.")
    rationale: str = Field(..., description="A brief description of your thought process that led you to the classification that you selected.")

model = ChatOllama(model="llama3.1:8b", temperature=0)

system_message = """
You are an assistant for ANDES, a library for power system modeling and simulation. your only job is to return the original message, plus the category that the message is. Do not provide any additional output in your response. Messages can be one of the following categories:

* question_general - An educational query about some aspect of power systems generally. These questions may not be specific to any aspect of ANDES. Examples include "What is a PV curve?", "Explain to me what eigenvalue analysis does", or "How do I interpret the results of a time-domain simulation?"
* question_param - An educational query about some aspect of an ANDES system loaded elsewhere. Examples include "Tell me about the system's PFlow member", "Analyze the final results of the simulation", or "What is the final result at each bus?"
* load_system - A command to load a new ANDES system from some path. Examples include "Load the test case kundur/kundur_full.xlsx" or "Load the system my_custom_setup.json from the current directory"
* interpreter - Instructions to perform some actions with an ANDES system. Any commands that do not belong in one of the previous categories should be placed into this category.
"""

print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "What is a power-flow simulation?"}]))
print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "What is the current value of the system's eigenvalues?"}]))
print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "Load a random test case and run a power-flow simulation"}]))
print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "Load a random test case and run a power-flow simulation"}]))

