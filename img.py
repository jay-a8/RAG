from langchain_community.document_loaders import UnstructuredImageLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import pytesseract

# 1. Use a Vision-Language Model.to process the image and provide embeddings.

# Load an image into the model (no text required).
# Generate an embedding that represents the visual content.
# Query the model to answer questions about the image.


# https://python.langchain.com/docs/integrations/document_loaders/image/
# https://blog.langchain.dev/semi-structured-multi-modal-rag/

# check if img exists
import os.path
print(os.path.isfile("bunny.jpg"))

# load the image
image_loader = UnstructuredImageLoader("bunny.jpg")
docs = image_loader.load()

# embedded the img
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "what is written in image?"
retrieved_docs = retriever.invoke(query)
print(retrieved_docs[0].page_content)
