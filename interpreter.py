import multiprocessing as mp
import os
import tempfile
from traceback import format_exception

import andes
import dill
import numpy as np
from langchain.messages import SystemMessage

from rag import retrieve_context

def codegen(state):
    pass # TODO: find previous prompt used to generate code
    
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

def interpreter(state):
    code = state["messages"][-1]["content"]

    q1, q2 = mp.Pipe(False)
    proc = mp.Process(target=interpreter_child, args=(dill.dumps(state["ss"]), code, q2))
    proc.start()
    
    new_ss = q1.recv()
    
    proc.join()
    
    update = {}
    
    if isinstance(new_ss, Exception):
        update["messages"] = [SystemMessage(content="Your code failed to run.\n" + "".join(format_exception(new_ss)))]
    else:
        update["messages"] = [SystemMessage(content="Your code ran successfully, ss has been updated.")]
        update["ss"] = dill.loads(new_ss)
        
    return update
