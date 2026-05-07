from langchain_ollama import ChatOllama
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from typing_extensions import Literal

class MessageClassifier(BaseModel):
    message: str = Field(..., description="The original message, verbatim, as supplied by the user")
    #category: Literal["question_general", "question_param", "interpreter"] = Field(..., description="A classification of the kind of sentence that the message is.")

model = ChatOllama(model="llama3.1:8b", temperature=0)

system_message = """
You are an assistant for ANDES, a library for power system modeling and simulation. your only job is to extract the first independent clause in the user's message. Do not attempt to reply to the original message. Do not provide any additional output in your response.
"""

print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "What is a power-flow simulation?"}]))

messages = [{"role": "system", "content": system_message}, {"role": "user", "content": "What is the current value of the system's eigenvalues?"}]
while True:
    x = model.invoke(messages)
    print(x)
    messages.append(x)
    
    if x.content == "What is the current value of the system's eigenvalues?":
        break
        
    messages.extend([{"role": "system", "content": "Your reply was incorrect. You are trying to reply to the original message, and that is explicitly what I told you not to do. Stop hallucinating a response and only extract the first independent clause."}, {"role": "user", "content": "What is the current value of the system's eigenvalues?"}])

print(model.with_structured_output(MessageClassifier).invoke([{"role": "system", "content": system_message}, {"role": "user", "content": "Load a random test case and run a power-flow simulation"}]))
