import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import itertools

def init_context(embedding):
    if init_context.vec_store is None:
        try:
            init_context.vec_store = FAISS.load_local("andes_doc_embeddings", embedding, allow_dangerous_deserialization=True)
        except RuntimeError:
            print("Creating vector store for ANDES manual, this could take a while...", file=sys.stderr)
            docs = PyPDFLoader("https://docs.andes.app/_/downloads/en/latest/pdf/").load()

            for d in docs:
                d.page_content = " ".join(line.strip() for line in d.page_content.split("\n") if line.strip())

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""])
            chunks = splitter.split_documents(docs)

            init_context.vec_store = FAISS.from_documents(chunks, embedding)
           init_context.vec_store.save_local("andes_doc_embeddings")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Look up information in the ANDES manual for a specific query."""
    retrieved_docs = init_context.vec_store.similarity_search(query, k=10)
    serialized = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return serialized, retrieved_docs
