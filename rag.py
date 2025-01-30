from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from numpy import dot
from numpy.linalg import norm


# https://python.langchain.com/v0.2/docs/tutorials/rag/
# https://python.langchain.com/v0.2/docs/tutorials/local_rag/

# 1. load documents
# 2. split documents into chunks
# 3. embed the chunks
# 4. stored the embeds into vector store
# 5. given a query, embed query, and find relevant embeded docs in the vectoratore 
# 6. asking query with relevant data to LLM model
# 

import os
print(os.path.exists("./data.txt"))  #check existence



# load text, encoding with utf-8
loader = TextLoader("./data.txt", encoding="utf-8")
docs = loader.load()


total_length = sum(len(doc.page_content) for doc in docs)
num_chunks = max(1, len(docs) * 5)
dynamic_chunk_size = max(100, total_length // num_chunks)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=dynamic_chunk_size,  chunk_overlap=0)

doc_list = text_splitter.split_documents(docs)

print(doc_list[0].page_content)



# print(doc_list[0])

embeddings = OllamaEmbeddings(model="llama3.2")

# save embedding and its text into vector store
vectorstore = InMemoryVectorStore.from_documents(
    doc_list,
    embedding=embeddings,
)



query = "who is mark"

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}) 

# Retrieve the most similar text
retrieved_documents = retriever.invoke(query)

# show the retrieved document's content
relavent_doc = retrieved_documents[0].page_content
print(relavent_doc)


query_embedding = embeddings.embed_query(query)
# print(query_embedding)


doc_embedding = embeddings.embed_query(relavent_doc)

cosine_similarity = dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))

print(f"similiarty between query and doc: {cosine_similarity:.4f}")

llm = OllamaLLM(model="llama3.2")
# system_prompt = (
#     "You are a consice AI, you will reply my question with my provided document."
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{context}\n\n{input}"), 
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# chain = create_retrieval_chain(retriever, question_answer_chain)

# response = chain.invoke({"input": query})
# print(response)

test = "You are a consice AI, you will reply my question with my provided data. This is the relavent documentation for this query, pls use it for your answer: " + relavent_doc + "This is the query:" + query
# print(test)
response = llm.invoke(test)
print(response)
