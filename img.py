from langchain_community.document_loaders import UnstructuredImageLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import pytesseract

# Set the Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# print(pytesseract.get_tesseract_version())


# https://python.langchain.com/docs/integrations/document_loaders/image/
# https://blog.langchain.dev/semi-structured-multi-modal-rag/

import os.path
print(os.path.isfile("bunny.jpg"))

image_loader = UnstructuredImageLoader("bunny.jpg")
docs = image_loader.load()

embeddings = OllamaEmbeddings(model="llama3")
vectorstore = FAISS.from_documents(docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is written in the image?"
retrieved_docs = retriever.invoke(query)
print(retrieved_docs[0].page_content)
