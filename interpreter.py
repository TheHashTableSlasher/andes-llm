import multiprocessing as mp
import os
from io import BytesIO
import subprocess
import sys
import socket
import warnings
from traceback import format_exception

import andes
import dill
import numpy as np
from langchain.messages import SystemMessage, HumanMessage
import docker

from rag import retrieve_context

__all__ = ["codegen", "interpreter"]

DOCKERFILE = '''
FROM python:{python_version}

RUN python3 -m pip install --break-system-packages dill numpy pandas git+https://github.com/CURENT/andes@v{andes_version}
RUN echo "{client_code}" > /main.py
RUN mkdir /pycode
'''

CLIENT_CODE = '''
import socket
import sys

import andes
import dill
import numpy as np
import pandas as pd

andes.system.import_pycode("/pycode")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP) as sock:
    sock.connect(("host.docker.internal", int(sys.argv[1])))
    
    data = bytearray()
    
    while True:
        chunk = sock.recv(0x10000)
        if not chunk:
            break
        data.extend(chunk)
    
    ss, code = dill.loads(data)
    global_variables = {"andes": andes, "np": np, "pd": pd, "ss": ss}
        
    try:
        exec(code, global_variables)
    except Exception as err:
        print(err)
        result = err
    else:
        result = global_variables["ss"]
   
    sock.sendall(dill.dumps(result))
    sock.shutdown(socket.SHUT_WR)
'''

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

def interpreter(model):
    python_version = ".".join(sys.version.split('.')[:2])
    andes_version = andes.__version__ if not andes.__version__.endswith("+unknown") else "2.0.0"
    pycode_path = os.path.dirname(andes.system.import_pycode().__file__)
    
    image = "andes-llm:{}_{}".format(python_version, andes_version)
    client = docker.from_env()
    
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        print("Creating docker image, this could take a while...", file=sys.stderr)
        
        dockerfile = DOCKERFILE.format(
            python_version = python_version,
            andes_version = andes_version,
            client_code = CLIENT_CODE.replace("\n", "\\n").replace("\"", "\\\"")
        )
        
        client.images.build(tag=image, fileobj=BytesIO(dockerfile.encode()))

    def closure(state):
        k = -1
        while not isinstance(state["messages"][k], SystemMessage):
            k -= 1
            
        messages = [model.invoke(state["messages"][k:])]
        code = messages[0].content
        
        if state.get("debug", False):
            print(f"\033[31minterpreter: will attempt to run the following Python code:\n{code}\033[0m")
            
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP) as sock:
            sock.bind(("", 0))
            sock.listen()
            
            port = sock.getsockname()[1]
        
            #client.containers.run(
            #    image, f"python3 /main.py {port}",
            #    extra_hosts={"host.docker.internal": "host-gateway"},
            #    volumes={pycode_path: {'bind': '/pycode', 'mode': 'ro'}},
            #    auto_remove=True,
            #    detach=True
            #)
            
            print("1 ({})".format(port))
            sock2 = sock.accept()[0]
            
            print("2")
            sock2.sendall(dill.dumps((state["ss"], code)))
            sock2.shutdown(socket.SHUT_WR)
            
            print("3")
            data = bytearray()

            while True:
                chunk = sock.recv(0x10000)
                if not chunk:
                    break
                data.extend(chunk)
            print("4")
            
            sock2.close()
            
        new_ss = dill.loads(data)
        
        if isinstance(new_ss, Exception):
            mesages.append(SystemMessage(content="Your code failed to run.\n" + "".join(format_exception(new_ss))))
            raise new_ss
        else:
            mesages.append(SystemMessage(content="Your p = subprocess.Popen(args)code ran successfully, ss has been updated."))
            update["ss"] = new_ss
            
        return {"messages": messages}
        
    return closure
