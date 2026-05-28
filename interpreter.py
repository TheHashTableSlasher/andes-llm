import multiprocessing as mp
import os
import tempfile
from traceback import format_exception

import andes
import dill
import numpy as np
from langchain.messages import SystemMessage, HumanMessage

from rag import retrieve_context

__all__ = ["codegen", "interpreter"]

def codegen(model):
    system_message = """
You are an assistant for ANDES, a library for power system modeling and simulation. Your job is to write code that performs the task the user is asking you to do. Only write the code - do not comment on or annotate it in any form, shape, or way. Use the tools at your disposal to search the ANDES user manual in order to get more information about the code you need to write before writing anything.
"""
    model = model.bind_tools([retrieve_context])

    def closure(state):
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=state["steps"][0])
        ]
        
        messages.append(model.invoke(messages))
        
        if state.get("debug", False):
            print(f"\033[31mcodegen: will attempt the following tool calls: \"{messages[-1].tool_calls}\"\033[0m")
        
        return {"messages": messages}
    
    return closure
    
def interpreter_child(dilled_ss, code, q):
    with tempfile.TemporaryDirectory() as tmproot:
        os.chroot(tmproot)
        
        variables = {
            "andes": andes,
            "np": np,
            "ss": dill.loads(dilled_ss)
        }
        
        try:
            exec(code, variables)
        except Exception as err:
            q.send(err)
        else:
            q.send(dill.dumps(variables["ss"]))

def interpreter(model):
    def closure(state):
        k = -1
        while not isinstance(state["messages"][k], SystemMessage):
            k -= 1
            
        messages = [model.invoke(state["messages"][k:])]
        code = messages[0].content
        
        if state.get("debug", False):
            print(f"\033[31minterpreter: will attempt to run the following Python code:\n{code}\033[0m")

        q1, q2 = mp.Pipe(False)
        proc = mp.Process(target=interpreter_child, args=(dill.dumps(state["ss"]), code, q2))
        proc.start()
        
        new_ss = q1.recv()
        
        proc.join()
        
        update = {"messages": messages}
        
        if isinstance(new_ss, Exception):
            mesages.append(SystemMessage(content="Your code failed to run.\n" + "".join(format_exception(new_ss))))
        else:
            mesages.append(SystemMessage(content="Your code ran successfully, ss has been updated."))
            update["ss"] = dill.loads(new_ss)
            
        return update
        
    return closure
