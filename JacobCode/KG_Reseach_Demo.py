from kg_builder import docs_to_kg
import json
import os
import networkx as nx
import pickle

import matplotlib.pyplot as plt


# This is an array of all documents you want to be put into the graph
# Works with .txt, .pdf, and .json of the format [{"image": <image name>, "description":<text description of image>}]
txt_file_paths = ["test.txt", "test2.txt"]

# This is the file you want the graph to be saved to
graph_file = "Test.json"

# This is the file where the chunked documents will be stored
pickle_file = "Test.pkl"


# Create Graph
G, chunks = docs_to_kg(txt_file_paths, graph_file, pickle_file)

nx.node_link_data(G, edges="edges")

# Visualize the graph
plt.figure(figsize=(24, 20))
pos = nx.spring_layout(G, k=0.5, iterations=100)
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=300, font_size=5)

# Write the relationships between nodes, gets ugly fast with large KGs
edge_labels = nx.get_edge_attributes(G, 'type')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


plt.title("Knowledge Graph Visualization")
plt.show()


from kg_builder import reasoning, KG_RAG, gpt_call
import json
import os
import networkx as nx
import pickle
from kg_builder import docs_to_kg



# This is the question you want to ask ChatGPT
query = "What does Mark do?" # simple queries may not have very good explainations as they are usually just lookups in the text

# This is the file you want the graph to be saved to
graph_file = "Test.json"

# This is the file where the chunked documents will be stored
pickle_file = "Test.pkl"



# Open the 
with open(graph_file, 'r') as f:
    graph_data = json.load(f)
        
G = nx.node_link_graph(graph_data)
nx.set_edge_attributes(G, 1, 'weight')

with open(pickle_file, 'rb') as file:
    chunks = pickle.load(file)

# Outputs all documents used from the most similar nodes
# Occasionally can be ALOT of documents
docs = KG_RAG(G, query, chunks)

#print(f"Documents: \n ")
#for doc in docs:
#    print(doc)
 
ans = gpt_call(query, docs[:10])

explanation, files_used, path = reasoning(G, chunks, query, ans)
print(f"Query:\n   {query}\n\nGPT Answer:\n   {ans}\n\nPath:\n   {path}\n\nExplanation:\n   {explanation}\n\nFiles Used: {files_used}")