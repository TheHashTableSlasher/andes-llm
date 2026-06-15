import sys
from zipfile import ZipFile

from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import HTMLSemanticPreservingSplitter

__all__ = ["init_context", "retrieve_context"]

def init_context(embedding):
    if init_context.vec_store is None:
        try:
            init_context.vec_store = FAISS.load_local("andes_doc_embeddings", embedding, allow_dangerous_deserialization=True)
        except RuntimeError:
            print("Creating vector store for ANDES manual, this could take a while...", file=sys.stderr)
            
            chunks = []
            splitter = HTMLSemanticPreservingSplitter(
                headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
                separators=["\n\n", "\n", ". ", "! ", "? "],
                max_chunk_size=500,
                preserve_images=False,
                preserve_videos=False,
                elements_to_preserve=["table", "ul", "ol", "code", "pre"],
                denylist_tags=["script", "style", "head"]
            )
            
            with ZipFile("andes_docs.zip") as zdir:
                for fileinfo in zdir.infolist():
                    if fileinfo.filename.endswith(".html"):
                        chunks.extend(splitter.split_text(zdir.read(fileinfo.filename)))

            init_context.vec_store = FAISS.from_documents(chunks, embedding)
            init_context.vec_store.save_local("andes_doc_embeddings")
            
init_context.vec_store = None

@tool(response_format="content_and_artifact")
def retrieve_context(query: str, k: int = 5):
    """Look up information in the ANDES manual for a specific query."""
    retrieved_docs = init_context.vec_store.similarity_search(query, k)
    serialized = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return serialized, retrieved_docs
