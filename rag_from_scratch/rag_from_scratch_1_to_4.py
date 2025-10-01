import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['OPENAI_API_KEY'] = "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"
os.environ['OLLAMA_API_KEY'] = "3430f7a1c798470db0765cd591be9df6.URvoPCDPvy6TxycAyz9nKttp"
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/"
#!/usr/bin/env python3
from bs4 import SoupStrainer
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import sentence_transformers


#### INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
                                    collection_name="rag-demo")

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = OllamaLLM(model="llama2", temperature=0)
# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")
# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string(question, "cl100k_base")


embd = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
len(query_result)

import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
                                    collection_name="rag-demo")

#retriever = vectorstore.as_retriever()
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  #k is the number of neighbors to search... 
docs = retriever.invoke("What is Task Decomposition?")
len(docs)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = Chattemplate = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

#chain = prompt | llm | StrOutputParser()
#chain.invoke({"context": docs[0].page_content, "question": "What is Task Decomposition?"})
prompt_hub_rag = hub.pull("rlm/rag-prompt")
#print(prompt_hub_rag.template)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain.invoke("What is Task Decomposition?")

