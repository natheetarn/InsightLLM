from flask import Flask
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate

from dotenv import load_dotenv
load_dotenv()
from flask import request
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
   
        
app = Flask(__name__)

@app.route("/query", methods=["GET"])
def query_index():
    global scb_index, kbank_index, ttb_index, siri_index, rerank, llm
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    kbank_engine = kbank_index.as_query_engine(similarity_top_k=10,node_postprocessors=[rerank])
    scb_engine = scb_index.as_query_engine(similarity_top_k=10,node_postprocessors=[rerank])
    siri_engine = siri_index.as_query_engine(similarity_top_k=10,node_postprocessors=[rerank])
    ttb_engine = ttb_index.as_query_engine(similarity_top_k=10,node_postprocessors=[rerank])
    
    from llama_index.core import PromptTemplate

    new_bank_templ_str= (
        "You are an expert in financial information"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge "
        "answer the query. You must always cite the page number. You always answer in Thai. You should answer as thoroughly as possible\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    new_bank_tmpl = PromptTemplate(new_bank_templ_str)
    scb_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_bank_tmpl,}
    )
    kbank_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_bank_tmpl}
    )
    siri_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_bank_tmpl,}
    )
    ttb_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_bank_tmpl}
    )

    query_engine_tools = [
    QueryEngineTool(
        query_engine=kbank_engine,
        metadata=ToolMetadata(
            name="kbank_agent",
            description=(
                "Provides information about kbank (kasikorn bank) (ธนาคารกสิกรไทย) financials "
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=scb_engine,
        metadata=ToolMetadata(
            name="scb_agent",
            description=(
                "Provides information about SCB Bank (ธนาคารไทยพานิชย์) financials. "
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=siri_engine,
        metadata=ToolMetadata(
            name="siri_agent",
            description=(
                "Provides information about Sansiri Real Estate(แสนสิริ) financials. "
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=scb_engine,
        metadata=ToolMetadata(
            name="ttb_agent",
            description=(
                "Provides information about TTB Bank (ธนาคารทหารไทย) financials. "
            ),
        ),
    ),
]
    agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose='debug',
    # context=context
)

    new_summary_tmpl_str = (
    """You are designed to help with a variety of tasks, 
    from answering questions to providing summaries to other types of analyses.\n\n
    ##Rules
        You must always attach page numbers to the information you referred to.
        if the user ask you to compare, you must use at least two tools.
        You must inpu the tools in Thai.
    ## Tools\n\nYou have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.\n
    This may require breaking the task into subtasks and using different tools to complete each subtask.\n\n
    You have access to the following tools:\n{tool_desc}\n\nHere is some context to help you answer the question and plan:\n{context}\n\n\n
    ## Output Format\n\nPlease answer in the same language as the question and use the following format:\n\n```\nThought: The current language of the user is: (user\'s language). I need to use a tool to help me answer the question.\nAction: tool name (one of {tool_names}) if using a tool.\nAction Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})\n
    ```\n\nPlease ALWAYS start with a Thought.\n\nPlease use a valid JSON format for the Action Input. Do NOT do this {{\'input\': \'hello world\', \'num_beams\': 5}}.\n\n
    If this format is used, the user will respond in the following format:\n\n```\nObservation: tool response\n```\n\n
    You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in the one of the following two formats:\n\n```\n
    Thought: I can answer without using any more tools. I\'ll use the user\'s language to answer\nAnswer: [your answer here (In the same language as the user\'s question)]\n```\n\n```\nThought: I cannot answer the question with the provided tools.\nAnswer: [your answer here (In the same language as the user\'s question)]\n
    ```\n\n## Current Conversation\n\nBelow is the current conversation consisting of interleaving human and assistant messages.\n
    """
    )
    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    agent.update_prompts({"agent_worker:system_prompt": new_summary_tmpl})
    response = agent.chat(query_text)
    return str(response), 200

@app.route("/")
def home():
    return "Hello World!"

if __name__ == "__main__":
    initialize_index()
    app.run(host="0.0.0.0", port=8000)