import heapq
import networkx as nx
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors
import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm

# Download the model Word2Vec
#model = api.load("word2vec-google-news-300")
#model.save_word2vec_format("word2vec-google-news-300.bin", binary=True)

#model = KeyedVectors.load_word2vec_format("word2vec-google-news-300.bin", binary=True)

#model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# Similarity score for embeddings
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Heuristic function for A*, cost between any 2 points
def heuristic(start, target):
    
    try:
        #Using word2Vec
        #similarity = model.similarity(start, target)
        #print(f"Similarity between {start} and {target}: {similarity:.4f}")
        similarity = cosine_similarity(start["embedding"], target["embedding"])
    except:
        # Default value when error occurs
        return 0.5
    
    # Cosine similarity seems identical to ^^^
    #vector1 = model[start]
    #vector2 = model[target]
    #cosine_similarity = dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    #print(f"Cosine Similarity between {start} and {target}: {cosine_similarity:.4f}")
    return 1 - similarity

def a_star_search(G, start, goal):
    s = G.nodes[start]
    g = G.nodes[goal]

    if s == None or g == None:
        raise Exception(f"Start or end node don't exist in graph. start: {start}, end: {goal}")

    # Priority queue for the open set
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Dictionaries for cost and path
    from_start_cost = {start: 0}
    total_cost = {start: heuristic(s, g)}
    came_from = {start: None}
    
    while open_set:
        current_cost, current_node = heapq.heappop(open_set)
        
        if current_node == goal:
            return reconstruct_path(came_from, start, goal)
        
        for neighbor in G.neighbors(current_node):
            tentative_from_start_cost = from_start_cost[current_node] + G[current_node][neighbor].get('weight', 1)
            
            if neighbor not in from_start_cost or tentative_from_start_cost < from_start_cost[neighbor]:
                from_start_cost[neighbor] = tentative_from_start_cost
                n = G.nodes[neighbor]
                total_cost[neighbor] = tentative_from_start_cost + heuristic(n, g)
                heapq.heappush(open_set, (total_cost[neighbor], neighbor))
                came_from[neighbor] = current_node
    
    return None

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def path_to_nodes_and_relationships(G, path):
    nodes_and_relationships = []
    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]
        relationship = G.get_edge_data(source, target)
        #print(relationship)
        nodes_and_relationships.append((source, relationship["type"], target))
    return nodes_and_relationships

# old function used to visually see the path in the graph, doesn't work well with large graphs
def visualize_graph_with_path(G, path):
    start = path[0]
    goal = path[-1]
    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    
    plt.figure(figsize=(12, 12))

    # Draw all nodes and edges in grey
    nx.draw(G, pos, with_labels=True, node_color='grey', edge_color='grey', node_size=500, alpha=0.6, font_size=10)
    
    # Highlight nodes and edges in the path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightblue', node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in path}, font_color='black', font_size=10)
    
    # Highlight start and goal nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='lime', node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color='red', node_size=700)
    
    plt.title("Knowledge Graph with Highlighted Path")
    plt.show()
