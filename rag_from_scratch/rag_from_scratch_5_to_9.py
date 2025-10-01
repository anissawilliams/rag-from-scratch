import os

# Configuration: prefer environment variables. It's safer to set these in your shell
# instead of hard-coding them in source control.
# LANGCHAIN_ENDPOINT and LANGCHAIN_API_KEY should be provided in the environment
LANGCHAIN_ENDPOINT = os.environ.get('LANGCHAIN_ENDPOINT')
LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY')
USER_AGENT = os.environ.get('USER_AGENT')

if not LANGCHAIN_ENDPOINT or not LANGCHAIN_API_KEY:
    # Not fatal here, but warn the user that LangSmith tracing won't work without these
    import warnings
    warnings.warn('LANGCHAIN_ENDPOINT or LANGCHAIN_API_KEY not set; LangSmith tracing will be disabled.')
else:
    # Enable tracing only when endpoint and API key are present to avoid unauthenticated requests
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = LANGCHAIN_ENDPOINT
    os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
#!/usr/bin/env python3
from bs4 import SoupStrainer
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
import sentence_transformers
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_openai import ChatOpenAI

#### INDEXING ####

# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
                                    collection_name="rag-demo")

retriever = vectorstore.as_retriever()

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_perspectives 
    | OllamaLLM(model="llama2", temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)



def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
# NOTE: we build the retrieval chain here; actual invocation is done in the demo below


# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = OllamaLLM(model="llama2", temperature=0)

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

# NOTE: final chain is constructed above; invocation is done in the demo under __main__

question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})
len(docs)
docs = loader.load()

# RAG-Fusion: Related
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results: list[list], k: int = 60):
    """
    Perform Reciprocal Rank Fusion (RRF) on multiple lists of ranked documents.

    This implementation uses a stable, hashable key for each document:
      - If the document has a metadata field `id` it will be used.
      - Otherwise a SHA1 over the document text and sorted metadata pairs is used.

    The function avoids serializing with `dumps`/`loads` and therefore runs much faster
    and works with unhashable document objects.

    Args:
        results: sequence of ranked lists (each element can be a Document-like object,
                 a dict with 'page_content' and 'metadata', or a plain string).
        k: RRF smoothing constant (default 60).

    Returns:
        List of tuples (original_doc, fused_score) sorted by fused_score desc.
    """
    import hashlib

    fused_scores: dict[str, float] = {}
    key_to_doc: dict[str, object] = {}

    def doc_key(doc) -> str:
        # If it's a mapping with metadata and id, prefer that
        try:
            meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else None)
        except Exception:
            meta = None

        # Prefer explicit id if present
        if isinstance(meta, dict) and "id" in meta and meta["id"] is not None:
            return str(meta["id"])

        # Otherwise use page_content if available
        try:
            text = getattr(doc, "page_content") if hasattr(doc, "page_content") else (doc.get("page_content") if isinstance(doc, dict) else str(doc))
        except Exception:
            text = str(doc)

        # Compute sha1 over text + stable metadata string
        meta_items = ""
        if isinstance(meta, dict):
            items = sorted((str(k), str(v)) for k, v in meta.items())
            meta_items = "||".join(f"{k}={v}" for k, v in items)

        h = hashlib.sha1()
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(meta_items.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    for docs in results:
        for rank, doc in enumerate(docs):
            key = doc_key(doc)
            # register original doc for final output (first-seen wins)
            if key not in key_to_doc:
                key_to_doc[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rank + k)

    # Build sorted results
    sorted_items = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(key_to_doc[k], score) for k, score in sorted_items]

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
# retrieval_chain_rag_fusion is constructed here; invocation is done in the demo below

# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# Run
question = "What are the main components of an LLM-powered autonomous agent system?"
#questions = generate_queries_decomposition.invoke({"question":question})
#print(questions)

#PART 6
# RAG-Fusion: Related
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

if __name__ == "__main__":
    # Demo run: only execute when run as a script (prevents heavy work on import)
    question = "What is task decomposition for LLM agents?"
    # build a retrieval chain using the (efficient) reciprocal_rank_fusion defined above
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    # run retrieval chains and the final RAG pipeline
    # 1) run the earlier retrieval_chain (generate_queries -> retriever.map -> get_unique_union)
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question": question})
    print("Unique union retrieval returned", len(docs), "documents")

    # 2) run RAG-fusion retrieval (generate_queries -> retriever.map -> reciprocal_rank_fusion)
    docs_rag_fusion = retrieval_chain_rag_fusion.invoke({"question": question})
    print("RAG-fusion retrieval returned", len(docs_rag_fusion), "documents")

    # 3) run final RAG chain which consumes the rag-fusion retrieval
    prompt = ChatPromptTemplate.from_template(template)
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Optional: instrument the run with LangSmith client (if installed) to add tags/metadata.
    try:
        from langsmith import Client as LangSmithClient
        client = LangSmithClient(api_key=LANGCHAIN_API_KEY) if LANGCHAIN_API_KEY else None
    except Exception:
        client = None

    if client:
        run_name = os.environ.get('LANGCHAIN_RUN_NAME', f'rag-demo-{int(__import__("time").time())}')
        # create a simple run metadata dict and log it (LangSmith SDK offers more features)
        try:
            run = client.create_run(name=run_name)
            # attach some metadata
            client.log_run_metadata(run.id, {"question": question, "script": "rag_from_scratch_5_to_9.py"})
        except Exception:
            # If any of this fails, proceed without failing the script
            pass

    #result = final_rag_chain.invoke({"question": question})
    #print("Final RAG result:\n", result)
    


# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# Run
question = "What are the main components of an LLM-powered autonomous agent system?"
questions = generate_queries_decomposition.invoke({"question":question})
print("Decomposition questions:", questions)

# Prompt
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

q_a_pairs = ""
for q in questions:
    
    rag_chain = (
    {"context": itemgetter("question") | retriever, 
     "question": itemgetter("question"),
     "q_a_pairs": itemgetter("q_a_pairs")} 
    | decomposition_prompt
    | llm
    | StrOutputParser())

    #rag_chain.invoke({"question": question})
    answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
    q_a_pair = format_qa_pair(q,answer)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    print(answer)
    
    examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
question = "What is task decomposition for LLM agents?"
#generate_queries_step_back.invoke({"question": question})

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

chain.invoke({"question": question})