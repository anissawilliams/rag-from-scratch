import os

from pydantic import BaseModel, Field
# from pyparsing import Optional  # Remove this line, use Optional from typing
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
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import sentence_transformers
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from typing import  Literal, Optional, Union
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import re

# Step 1: Define your structured output model


class RouteQuery(BaseModel):
    origin: str
    destination: str
    mode: str


# Step 2: Create the parser
parser = PydanticOutputParser(pydantic_object=RouteQuery)

# Step 3: Create a strong prompt with example and strict instructions
prompt = PromptTemplate.from_template(
    "Extract route info from this request:\n\n{request}\n\n"
    "Respond ONLY with a JSON object using these exact keys: origin, destination, mode."
)




# Step 4: Create the LLM
llm = OllamaLLM(model="llama2", temperature=0)
formatted_prompt = prompt.format(request="I want to travel from New York to Boston by train")

chain = prompt | llm

import re
import json
from pydantic import ValidationError

def run_chain(request_text):
    raw = chain.invoke({"request": request_text})
    print("🔎 Raw output:\n", raw)

    # Extract JSON block
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        print("No JSON found.")
        return None

    try:
        data = json.loads(match.group(0))
        result = RouteQuery(**data)
        print("Parsed result:", result)
        return result
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        print("Parsing failed:", e)
        print("Extracted JSON:\n", match.group(0))
        return None

run_chain("I want to go from New York to Boston by train")