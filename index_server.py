import os

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings


from llama_index.core.tools import QueryEngineTool, ToolMetadata

# NOTE: for local testing only, do NOT deploy with your key hardcoded

index = None

scb_index = None
kbank_index = None
ttb_index = None
siri_index = None
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
def initialize_index():
    global scb_index, kbank_index, ttb_index, siri_index, llm
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/kbank"
        )
        kbank_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/scb"
        )
        scb_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/siri"
        )
        siri_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/ttb"
        )
        ttb_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False
    llm = Gemini(model="models/gemini-pro")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# global
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 128
    if not index_loaded:
    # load data
        kbank_docs = SimpleDirectoryReader(
            input_files=["kbank.PDF"]
        ).load_data()
        scb_docs = SimpleDirectoryReader(
            input_files=["scb.pdf"]
        ).load_data()
        siri_docs = SimpleDirectoryReader(
            input_files=["siri.PDF"]
        ).load_data()
        ttb_docs = SimpleDirectoryReader(
            input_files=["ttb.pdf"]
        ).load_data()

    # build index
        kbank_index = VectorStoreIndex.from_documents(kbank_docs)
        scb_index = VectorStoreIndex.from_documents(scb_docs)
        siri_index = VectorStoreIndex.from_documents(siri_docs)
        ttb_index = VectorStoreIndex.from_documents(ttb_docs)

        # persist index
        kbank_index.storage_context.persist(persist_dir="./storage/kbank")
        scb_index.storage_context.persist(persist_dir="./storage/scb")
        siri_index.storage_context.persist(persist_dir="./storage/siri")
        ttb_index.storage_context.persist(persist_dir="./storage/ttb")


    rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=3)
   
        