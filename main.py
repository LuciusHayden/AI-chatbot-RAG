import os
import dotenv

dotenv.load_dotenv('.env')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API")

print(os.getenv("OPENAI_API_KEY"))
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain import hub

from langgraph.graph import START, StateGraph

from typing_extensions import List, TypedDict


model = ChatOpenAI(model="gpt-4o-mini")

prompt = hub.pull("rlm/rag-prompt")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)


file_path = "data\Competition Manual - V6.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

doc_chunks = text_splitter.split_documents(docs)

print(len(doc_chunks))

ids = vector_store.add_documents(doc_chunks)

class State(TypedDict):
    question: str
    answer: str
    context: List[Document]


def retrieve(state: State):
    retrieved_docs =  vector_store.similarity_search(state["question"])
    print(retrieved_docs)
    return {'context' : retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {'answer': response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "How many points do I lose for a major foul?"})
print(response["answer"])