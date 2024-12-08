import os
import dotenv

dotenv.load_dotenv('.env')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API")

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain import hub
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver


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


graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs =  vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join((f"Source: {doc.metadata['source']} Content: {doc.page_content}") for doc in retrieved_docs)
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    model_tools = model.bind_tools([retrieve])
    response = model_tools.invoke(state['messages'])
    return {"messages": response}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate answer."""
    recent_messages = []
    for message in reversed(state['messages']):
        if message.type == "tool":
            recent_messages.append(message)
        else:
            break
    tool_messages = recent_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message for message in state["messages"]
        if message.type in ('human', 'system')
        or (message.type == 'ai' and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = model.invoke(prompt)
    return {"messages" : [response]}

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id" : "abc123"}}


prompt = "How many points do I get for a third level ascent?"


response = graph.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config=config
)

print(response)