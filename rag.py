from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# 1. load and split
# 2. embed and vector store
# https://python.langchain.com/v0.2/docs/tutorials/rag/
# https://python.langchain.com/v0.2/docs/tutorials/local_rag/

loader = TextLoader("./data.txt")
list = loader.load_and_split()

# for x in list:
#     print(x)
#     print("============================================================")

embeddings = OllamaEmbeddings(model = "llama3")
vector_store = InMemoryVectorStore(embeddings)

question = "What are the approaches to Task Decomposition?"
docs = vector_store.similarity_search(question)
len(docs)

