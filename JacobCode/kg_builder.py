from openaicom import convert_to_kg
from path_algos import a_star_search, path_to_nodes_and_relationships
from langchain_community.document_loaders import PyPDFLoader
from transformers import LlamaTokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from requests.exceptions import ChunkedEncodingError


import numpy as np
import networkx as nx
import json
import re
import time
import os
import matplotlib.pyplot as plt
import pickle

# def set_api_key(key):
#     api_key(key)

def chunk_file(file):
    if os.path.splitext(file)[1] == ".pdf":
        return chunk_text(file)
    elif os.path.splitext(file)[1] == ".json":
        with open(file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects.")

        documents = []
        for entry in data:
            if 'image' in entry and 'description' in entry:
                formatted_string = f"Image: {entry['image']}\nDescription: {entry['description']}"
                documents.append(Document(page_content=formatted_string))
            else:
                print(f"Skipping entry without 'image' or 'description': {entry}")
        return documents
    else:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Step 3: Create function to count tokens
        tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 1024,
            chunk_overlap  = 32,
            length_function = count_tokens,
        )

        return text_splitter.create_documents([text])

def chunk_text(document):
    # Advanced method - Split by chunk

    # Step 1: Convert PDF to text
    import textract
    doc = textract.process(document)

    # Step 2: Save to .txt and reopen (helps prevent issues)
    with open("temp.txt", 'w') as f:
        f.write(doc.decode('utf-8'))

    with open("temp.txt", 'r') as f:
        text = f.read()

    # Step 3: Create function to count tokens
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # Step 4: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1024,
        chunk_overlap  = 32,
        length_function = count_tokens,
    )

    return text_splitter.create_documents([text])

def docs_to_kg(txt_file_paths, file_path, pickle_path):
    for path in txt_file_paths:
        if not os.path.isfile(path):
            print(f"CANNOT FIND FILE {path}")

    kg = []

    text_chunks = {}
    for txt_file_path in txt_file_paths:
        paragraphs = chunk_file(txt_file_path)
        i = 0
        size = len(paragraphs)
        for paragraph in paragraphs:
            i += 1
            print(f'Converting chunk {i}/{size} in {txt_file_path}')
            k = convert_to_kg_plus(paragraph, i, txt_file_path)
            if k is None:
                continue
            # print(k)
            kg.append(k)
        text_chunks[txt_file_path] = paragraphs

    G = nx.DiGraph()
    for graph in kg:
        if graph is None:
            continue
        # Convert the JSON string to a Python dictionary if it's a string
        if isinstance(graph, str):
            try:
                graph = json.loads(graph)
            except json.JSONDecodeError as e:
                print(f"docstokg: {e}")
                continue

        append_to_graph(G, graph)

    # Convert graph to undirected
    G = G.to_undirected()

    update_graph_with_embeddings(G, text_chunks)
    
    # Save the graph to a JSON file
    graph_data = nx.node_link_data(G)
    #print(graph_data)
    with open(file_path, 'w') as f:
        json.dump(graph_data, f, indent=4)
    with open(pickle_path, 'wb') as file:
        pickle.dump(text_chunks, file)

    return G, text_chunks

def append_to_graph(graph, data):
    # Add nodes (entities) to the graph, along with metadata
    for entity in data['entities']:
        node_id = entity['id']
        
        # Extract file_name and paragraph_number to only store them in the 'references' map
        file_name = entity['file_name']
        paragraph_number = entity['chunk_number']
        
        # Check if the node already exists in the graph
        if graph.has_node(node_id):
            # If the node exists, update its references map
            node_data = graph.nodes[node_id]
            references = node_data.get('references', {})
            
            # Update the reference map with the current file and paragraph number
            if file_name in references:
                references[file_name].append(paragraph_number)
            else:
                references[file_name] = [paragraph_number]
            
            # Set the updated references map
            node_data['references'] = references
        else:
            # If the node doesn't exist, create it with metadata including references
            references = {file_name: [paragraph_number]}
            entity['references'] = references
            
            # Add the node with its metadata (without redundant file_name and paragraph_number)
            graph.add_node(node_id, **entity)
    
    # Add edges (relationships) to the graph
    for relationship in data['relationships']:
        source = relationship['source']
        target = relationship['target']
        graph.add_edge(source, target, **relationship)  # Add relationship metadata as edge attributes

def create_embedding_for_node(references, loaded_text_chunks):
    """Generate an embedding based on the references and the loaded text chunks."""
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    # Collect the relevant text chunks for this node
    node_texts = []
    for file_name, reference in references.items():
        for ref in reference:
            chunk = loaded_text_chunks[file_name][int(ref) - 1]
            node_texts.append(chunk.page_content)
    
    # Concatenate all text chunks into one single string
    concatenated_text = " ".join(node_texts)
    
    if not concatenated_text:
        return None  # In case the node has no valid references
    
    # Use OllamaEmbeddings to generate the embedding
    embedding = embedding_model.embed_query(concatenated_text)
    return np.array(embedding)  # Convert the embedding to a NumPy array for easy similarity calculations later

def update_graph_with_embeddings(G, loaded_text_chunks):
    """Update every node in the graph with an embedding based on its references."""
    for node, references in G.nodes(data='references'):
        if references is None:
            continue
        
        # Generate the embedding for the node based on the references
        node_embedding = create_embedding_for_node(references, loaded_text_chunks)
        
        if node_embedding is not None:
            # Add the embedding as an attribute to the node
            G.nodes[node]['embedding'] = node_embedding.tolist()
            print(f"Added embedding to {node}.")
        else:
            print(f"Node {node} has no valid references or failed to generate embedding.")

        
def split_into_paragraphs(text):
    # Split the text into paragraphs based on two or more newlines
    paragraphs = re.split(r'\n{2,}', text.strip())
    
    # Join lines within each paragraph to treat them as one
    paragraphs = [paragraph.replace('\n', ' ') for paragraph in paragraphs]
    
    return paragraphs

def convert_to_kg_plus(paragraph, paragraph_num, file_name, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            k = convert_to_kg(paragraph)
            # Attempt to load the JSON
            data = json.loads(k)
            for entity in data['entities']:
                entity['file_name'] = file_name
                entity['chunk_number'] = paragraph_num
                #entity['chunk'] = paragraph
            return data

        except ChunkedEncodingError as e:
            retries += 1
            print(f"ChunkedEncodingError occurred. Retry {retries}/{max_retries}.")
            # time.sleep(retry_delay)
            
        except json.JSONDecodeError as e:
            # If there's a JSON error, print a helpful error message and return None
            print(f"plus JSON for paragraph {paragraph_num}: {e}")
            return None

    print("Max retries reached. Could not complete the request, moving on.")
    return None

# Function to search nodes by a keyword and return chunk (paragraph) numbers
def search_nodes_by_keyword(graph, keyword):
    # List to store matching nodes and their associated chunk numbers
    matching_nodes = []

    # Iterate through all the nodes in the graph
    for node, data in graph.nodes(data=True):
        # Check if the keyword is in the node ID or any of its metadata
        #print(node)
        if keyword.lower() in node.lower():
            references = data.get('references')
            matching_nodes.append((node, references))
    
    return matching_nodes

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def search_nodes_by_embedding(graph, embedding, top_k=5):
    # List to store nodes with their similarity scores
    node_similarities = []

    # Iterate through all the nodes in the graph
    for node, data in graph.nodes(data=True):
        node_embedding = data.get('embedding')
        
        # Skip nodes that don't have embeddings
        if node_embedding is None:
            continue
        
        # Calculate cosine similarity between query embedding and node's embedding
        similarity = cosine_similarity(embedding, node_embedding)
        
        # Add the node, its references, and similarity score to the list
        references = data.get('references')
        node_similarities.append((node, references, similarity))
    
    # Sort the nodes by similarity score in descending order
    node_similarities = sorted(node_similarities, key=lambda x: x[2], reverse=True)
    
    # Return the top 5 most similar nodes
    return node_similarities[:top_k]

def search_graph(graph, query):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Use OllamaEmbeddings to generate the embedding
    embedding = embedding_model.embed_query(query)
    node_similarities = []
    # Iterate through all the nodes in the graph
    for node, data in graph.nodes(data=True):
        node_embedding = data.get('embedding')
        
        # Skip nodes that don't have embeddings
        if node_embedding is None:
            continue
        
        # Calculate cosine similarity between query embedding and node's embedding
        similarity = cosine_similarity(embedding, node_embedding)
        
        # Add the node, its references, and similarity score to the list
        references = data.get('references')
        node_similarities.append((node, data, similarity))
    node_similarities = sorted(node_similarities, key=lambda x: x[2], reverse=True)

    return node_similarities

def get_all_relationships(graph, nodes):
    relationships = set()  

    for node in nodes:
        for target, edge_data in graph[node].items():
            relationships.add((node, target, frozenset(edge_data.items())))

        # for directed graphs
        #for source, edge_data in graph.pred[node].items():   
        #    relationships.add((source, node, frozenset(edge_data.items())))

    return [{"source": r[0], "target": r[1], "relationship": dict(r[2])} for r in relationships]


def reasoning(G, chunks, query, ans):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    embedding_query = embedding_model.embed_query(query)
    embedding_ans = embedding_model.embed_query(ans)

    nodes_query = search_nodes_by_embedding(G, embedding_query)
    nodes_ans = search_nodes_by_embedding(G, embedding_ans)

    if(nodes_query is None or nodes_ans is None):
        return None

    nodes_comb = nodes_query + nodes_ans

    path = a_star_search(G, nodes_query[0][0], nodes_ans[1][0]) # pass in names of nodes
    relation_path = path_to_nodes_and_relationships(G, path)

    relationships = get_all_relationships(G, path)

    #print(f"Relationships: {relationships}")
    #print(f"path: {path}")
    #print(f"path with relations: {relation_path}")
        
    r_doc = Document(page_content=str(relationships))
    #path_doc = Document(page_content=str(relation_path))
    docs = [r_doc]
    
    #print(f"relations: {relationships}")
    #print(f"path: {path}")

    files_used = set() # used to for user to know which documents where used
    doc_used = set() # used to remove duplicates

    for node, references, _ in nodes_comb:
        #print(f"Node: {node}\n  reference: {references}")
        if references is None:
            continue

        for file_name, reference in references.items():
            files_used.add(file_name)
            for ref in reference:
                #try:
                doc_content = chunks[file_name][int(ref)-1]
                doc_content_str = str(doc_content)
                if doc_content_str not in doc_used:
                    doc_used.add(doc_content_str)
                    docs.append(Document(page_content=doc_content) 
                            if isinstance(doc_content, str) 
                            else doc_content
                            )
                #except:
                #    print(f"ERROR: failed to access reference: {ref} in text: {file_name}")
                #    continue
    
    return reasoning_gpt_call(query, ans, docs[:10], relation_path), files_used, relation_path # limit number of docs (arbitrarily chosen)
        

def KG_RAG(graph, query, chunks):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Use OllamaEmbeddings to generate the embedding
    embedding = embedding_model.embed_query(query)
    

    node_similarities = search_nodes_by_embedding(graph, embedding)

    kgdocs = []

    files_used = set() # used to for user to know which documents where used
    doc_used = set() # used to remove duplicates

    for node, references, _ in node_similarities:
        #print(f"Node: {node}\n  reference: {references}")
        if references is None:
            continue

        for file_name, reference in references.items():
            files_used.add(file_name)
            for ref in reference:
                try:
                    doc_content = chunks[file_name][int(ref)-1]
                    doc_content_str = str(doc_content)
                    if doc_content_str not in doc_used:
                        doc_used.add(doc_content_str)
                        kgdocs.append(Document(page_content=doc_content) 
                                if isinstance(doc_content, str) 
                                else doc_content
                                )
                except:
                    print(f"ERROR: failed to access reference: {ref} in text: {file_name}")
                    continue

    return kgdocs

def gpt_call(question, docs):
    # way 1: (load_qa_chain deprecated )
    # llm = OllamaLLM(model="llama3.2")  
    # chain = load_qa_chain(llm=llm, chain_type="stuff")
    # answer = chain.invoke({"input_documents": docs, "question": question})
    # if isinstance(answer, dict) and "output_text" in answer:
    #     answer = answer["output_text"]
    # print(answer)
    # return answer

    # way 2
    llm = OllamaLLM(model="llama3.2")
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the following question based on the provided documents:\nDocuments: {documents}\n\nQuestion: {question}\nAnswer:"
    )
    document_text = "\n".join([doc.page_content for doc in docs])
    chain = prompt | llm
    answer = chain.invoke({"documents": document_text, "question": question})
    if isinstance(answer, dict) and "output_text" in answer:
        answer = answer["output_text"]
    return answer


def reasoning_gpt_call(question, answer, docs, path):
    llm = OllamaLLM(model="llama3.2")

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant. Using the context given, provide a detailed explanation and reasoning to the question/answer, given a specific path of reasoning.

    Context:
    {context}

    Question: {question}
    Answer: {answer}
    Path: {path}

    Explanation:
    """)
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question, "answer": answer, "path": path})

    if isinstance(response, dict) and "text" in response:
        return response["text"]
    return response
