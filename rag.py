from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy import dot
from numpy.linalg import norm

#github: https://github.com/jay-a8/RAG

# https://python.langchain.com/v0.2/docs/tutorials/rag/
# https://python.langchain.com/v0.2/docs/tutorials/local_rag/

# 1. load documents
# 2. split documents into chunks
# 3. embed the chunks
# 4. stored the embeds into vector store
# 5. given a query, embed query, and find relevant embeded docs in the vectoratore 
# 6. asking query with relevant data to LLM model

import os
# print(os.path.exists("./data.txt"))  #check existence

# load text, encoding with utf-8
loader = TextLoader("./data.txt", encoding="utf-8")
docs = loader.load()

# decide token size
total_length = sum(len(doc.page_content) for doc in docs)
chunk_size = min(500, max(100, total_length // (len(docs) * 3)))

# split doc to tokens
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
doc_list = text_splitter.split_documents(docs)

# print(doc_list[0].page_content)

# save embedding and its text into vector store
# embedding returns the same numbers
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = InMemoryVectorStore.from_documents(doc_list, embedding=embeddings)

# Use the vectorstore as a retriever, and set the alg to consin similarity
# set to returns back one most relavant doc to the query
# this is the problem, the similarity value changes for documents
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) 

# query = "who is mark?"
# results = vectorstore.similarity_search_with_score(query, k=5)  # Retrieve top 5 documents

# for doc, score in results:
#     print(f"Document: {doc.page_content}\nSimilarity Score: {score}\n")

# ==================================================================
# A better way to retrive and answer the query
llm = OllamaLLM(model="llama3.2")

custom_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the provided context to answer the question concisely.

    Context:
    {context}

    Question: {question}
    Answer:
    """
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": custom_prompt})

query = "Who is Mark?"
response = qa_chain.invoke(query)
print(response["result"])



# ==============================================================
# # A way to retreive and answer the query, but this is for testing the difference between with and without the RAG

# query = "Who is Mark?"
# retrieved_documents = retriever.invoke(query) 

# # show the retrieved document's content
# relavent_doc = retrieved_documents[0].page_content


# llm = OllamaLLM(model="llama3.2")
# query = "Who is Mark?"
# test = "You are a consice AI, you will reply my question with my provided data. This is the relavent documentation for this query, pls use it for your answer: " + relavent_doc + "This is the query:" + query
# # response = llm.invoke(query)
# response = llm.invoke(test)
# temperture, topp

# print(relavent_doc)
# print("================")
# print(response)


# check cosine_simlarity with query and relavent doc
# query_embedding = embeddings.embed_query(query)
# doc_embedding = embeddings.embed_query(retrieved_documents[0].page_content)
# cosine_similarity = dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))
# print(f"similiarty between query and doc: {cosine_similarity:.4f}")