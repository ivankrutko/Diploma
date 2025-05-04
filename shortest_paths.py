import sys
import osmnx
import time as t
import random
import networkx as nx
import matplotlib.pyplot as plt

G = osmnx.graph_from_point((50.3869524, 30.4807769), dist=920)

s =  960023629
d =  334465761 

osmnx.plot.plot_graph(G, node_size=2, node_color='r', edge_color='b', edge_linewidth=0.2, bgcolor='white')

def graph_from_osmnx(G):
    graph = dict()
    for n, adj in G.adjacency():
        if n not in graph:
            graph[n] = dict()
        for e, eattr in adj.items():
            for _, iattr in eattr.items():
                if e not in graph[n] or graph[n][e] > iattr["length"]:
                    graph[n][e] = float(iattr["length"])
    return graph

graph={
  'A':{'B':2, 'C':4},
  'B':{'A':2, 'C':3,'D':8},
  'C':{'A':4,'B':3, 'E':5,'D':2},
  'D':{'B':8,'C':2, 'E':11,'F':22},
  'E':{'C':5, 'D':11,'F':1},
  'F':{'D':22, 'E':1},
}

source='A'
destination='F'

def node_data_map(graph):
    inf = sys.maxsize
    map={}
    for i in graph:
        map.update({i: {'cost':inf, 'pred':[]}}) 
    return map

def dijkstra(graph, source,destination):
    shortest_distance={}
    track_predecessor={}
    unseenvertices={}
    for v in graph:
        unseenvertices[v]=graph[v]
    infinity=sys.maxsize
    track_path=[]
    for node in unseenvertices:
        shortest_distance[node]=infinity
    shortest_distance[source]=0
    while unseenvertices:
        min_distance_node = None
        for node in unseenvertices:
            if min_distance_node is None:
                min_distance_node = node
            elif shortest_distance[node]<shortest_distance[min_distance_node]:
                min_distance_node=node
        path_options= graph[min_distance_node].items()   
        for child_node, weight in path_options:
            if weight + shortest_distance[min_distance_node]<shortest_distance[child_node]:
                shortest_distance[child_node]=weight + shortest_distance[min_distance_node]
                track_predecessor[child_node]=min_distance_node
        unseenvertices.pop(min_distance_node)
    currentNode = destination 
    while currentNode!=source:
        try:
            track_path.insert(0,currentNode)
            currentNode=track_predecessor[currentNode]
        except KeyError:
            ("Path is not reachable")
            break
    track_path.insert(0,source)

    if shortest_distance[destination]!=infinity:
        return shortest_distance,track_path, track_predecessor
    else:
        return shortest_distance,track_path
    
print("Dijkstra's algorithm:")
shortest_distance,track_path,predecessors=dijkstra(graph,source,destination)
print("Shortest distance is "+str(shortest_distance[destination]))
print("Optimal path is "+str(track_path))

graph=graph_from_osmnx(G)
t1 = t.time()
shortest_distance,track_path,predecessors=dijkstra(graph, s,d)
print("Shortest distance is "+str(shortest_distance[d]))
print("Optimal path is "+str(track_path))
t2 = t.time()
print('serial: ',t2 - t1, 's')
osmnx.plot.plot_graph_route(G, track_path, route_color='g', node_size=2, node_color='r', edge_color='b', edge_linewidth=0.2, bgcolor='white');

graph2={
    'A':{'B':6,'C':4,'D':5},
    'B':{'E':-1},
    'C':{'B':-2, 'E':3},
    'D':{'C':-2,'F':-1},
    'E':{'F':3},
    'F':{}
}

source='A'
destination='F'
def bellmanford(graph, source, destination):
    inf = sys.maxsize
    node_data=node_data_map(graph)
    node_data[source]['cost']=0
    for i in range(len(graph)-1):
        #print('Iteration '+ str(i))
        for itr in graph:
            for neighbor in graph[itr]:
                cost=node_data[itr]['cost']+graph[itr][neighbor]
                if cost<node_data[neighbor]['cost']:
                    if node_data[neighbor]['cost']==inf:
                        node_data[neighbor]['pred']=node_data[itr]['pred']+[itr]
                    else:
                        node_data[neighbor]['pred'].clear()
                        node_data[neighbor]['pred']=node_data[itr]['pred']+[itr]
                    node_data[neighbor]['cost']=cost
        #print(node_data)
    return node_data

print("Bellman-Ford algorithm:")
node_data=bellmanford(graph2, source,destination)
print('Shortest distance: '+str(node_data[destination]['cost']))
print('Shortest path: '+str(node_data[destination]['pred']+ [destination]))

t1 = t.time()
node_data=bellmanford(graph_from_osmnx(G), s,d)
print('Shortest distance: '+str(node_data[d]['cost']))
print('Shortest path: '+str(node_data[d]['pred']+ [d]))
t2 = t.time()
print('serial: ',t2 - t1, 's')

def printmatrix(m):
    r,c= len(m), len(m[0])
    for i in range(r):
        for j in range(c):
            print(m[i][j], end=" ")
        print()

def floydwarshall(graph):
    inf = float(sys.maxsize)
    vertices = []
    for i in graph:
        vertices.append(i)
    dist=[]
    for i in vertices:
        r=[]
        for j in vertices:
            if i==j:
                r.append(0.0)
            elif j in graph[i]:
                r.append(graph[i][j])
            else:
                r.append(inf)
        dist.append(r)
    n=len(dist)

    # Initialize predecessor (previous_node) matrix
    previous_node = [[None] * n for _ in range(n)]
    # Initialize previous_node for direct edges
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] != inf:
                previous_node[i][j] = vertices[i]  # Previous node is the source node

    # Main Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    previous_node[i][j] = previous_node[k][j]  # Update to reflect the predecessor

    # Check for negative cycles
    for i in range(n):
        if dist[i][i] < 0:
            print("Negative cycle detected.")
            return None, None
    return vertices, dist, previous_node

def reconstruct_path(vertices, previous_node, start, end):
    path = []
    current = end
    while current is not None and current != start:
        path.append(current)
        current = previous_node[vertices.index(start)][vertices.index(current)]
    if current is None:
        print(f"No path from {start} to {end}.")
        return
    path.append(start)
    path.reverse()
    print(path)

print("Floyd-Warshall algorithm:")
vertices, dist, previous_node = floydwarshall(graph2)
reconstruct_path(vertices,previous_node, source, destination)

t1 = t.time()
vertices, dist, previous_node = floydwarshall(graph_from_osmnx(G))
reconstruct_path(vertices,previous_node, s,d)
t2 = t.time()
print('serial: ',t2 - t1, 's')

def johnson(graph):
    dict_of_vertices={}
    for v in graph:
        dict_of_vertices[v]=0
    graph['additional_source']=dict_of_vertices
    new_source='additional_source'
    node_data=bellmanford(graph, new_source, list(graph.keys())[0])
    new_graph=graph
    for i in new_graph:
        if i!='additional_source':
            for j in new_graph[i]:
                new_graph[i][j]=node_data[i]['cost']+new_graph[i][j]-node_data[j]['cost']
    new_graph.pop('additional_source')
    paths={}

    for i in new_graph:
        shortest_distance,path,predecessors=dijkstra(new_graph, i, i)
        paths.update({(i,i): path}) 
        for j in new_graph:
            path=[]
            if i!=j:
                currentNode = j 
                while currentNode!=i:
                    try:
                        path.insert(0,currentNode)
                        currentNode=predecessors[currentNode]
                    except KeyError:
                        ("Path is not reachable")
                        break
                path.insert(0,i)
            paths.update({(i,j): path}) 
    return paths

test_graph={
    0:{1:4,4:1},
    1:{},
    2:{3:-2, 1:7},
    3:{1:1},
    4:{3:-5},
}

print("Johnson's algorithm")
paths=johnson(graph2)
print(paths)
print(paths[(source,destination)])

t1 = t.time()
paths=johnson(graph_from_osmnx(G))
t2 = t.time()
print('serial: ',t2 - t1, 's')
print(paths[(s,d)])

example_graph={
  1:{2:4, 3:6},
  2:{4:6, 3:2,5:4},
  3:{2:3,5:3},
  4:{5:8},
  5:{2:7,4:2},
}

example_graph2={
  1:{2:3, 3:5},
  2:{4:4, 3:7,5:-2},
  3:{4:-3,5:6},
  4:{2:-1},
  5:{4:5},
}

def plot_graph(graph, path):
    shortest_path_edges=[]
    i=0
    while(i<len(path)-1):
        shortest_path_edges.append((path[i],path[i+1]))
        i+=1
    print(shortest_path_edges)
    edge_list1=[]
    edge_list2=[]
    G = nx.DiGraph()
    for i in graph:
        for j in graph[i]:
            if graph.get(j).get(i)==None:
                G.add_edge(i, j, weight=graph[i][j])
                edge_list1.append((i,j))
            else:
                G.add_edge(i, j, weight=graph[i][j])
                edge_list2.append((i,j))
            
    # explicitly set positions
    pos = {1: (0, 0), 2: (2, 1), 3: (2, -1), 4: (4, 1), 5: (4, -1)}
    nx.draw_networkx_nodes(G, pos,
        node_size= 500,
        node_color="lightgray",
        edgecolors= "black")
    nx.draw_networkx_nodes(G, pos, nodelist=path,
        node_size= 500,
        node_color="lightgreen",
        edgecolors= "black")
    nx.draw_networkx_labels(G, pos, font_size = 14)
    nx.draw_networkx_edges(G, pos,edgelist=edge_list1, connectionstyle="arc3,rad=0")
    nx.draw_networkx_edges(G, pos,edgelist=edge_list2, connectionstyle="arc3,rad=0.2")
    for i in shortest_path_edges:
        if i in edge_list1:
            nx.draw_networkx_edges(G, pos,edgelist=[i],edge_color='lightgreen', connectionstyle="arc3,rad=0")
        else:
            nx.draw_networkx_edges(G, pos,edgelist=[i],edge_color='lightgreen', connectionstyle="arc3,rad=0.2")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,connectionstyle="arc3,rad=0.2")
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

source=1
destination=5
shortest_distance,track_path,predecessors=dijkstra(example_graph,source,destination)
print("Shortest distance is "+str(shortest_distance[destination]))
print("Optimal path is "+str(track_path))
plot_graph(example_graph, track_path)

node_data=bellmanford(example_graph2,source,destination)
print('Shortest distance: '+str(node_data[destination]['cost']))
print('Shortest path: '+str(node_data[destination]['pred']+ [destination]))
plot_graph(example_graph2, node_data[destination]['pred']+ [destination])

vertices, dist, previous_node = floydwarshall(example_graph)
reconstruct_path(vertices,previous_node, source, destination)

paths=johnson(example_graph)
print(paths)
print("Shortest distance is "+str(paths[(source,destination)]))

d1=0.1 
d2=0.5
d3=0.9

v=600

random.seed(111)

def edges_by_density(degree, density):
    return density*degree*(degree-1)

def modded_dict_of_dicts(g):
    dict_of_dicts=nx.to_dict_of_dicts(g)
    for i in dict_of_dicts:
        for j in dict_of_dicts[i]:
            dict_of_dicts[i][j]=random.randint(0,100)
    return dict_of_dicts

g1=modded_dict_of_dicts(nx.gnp_random_graph(v,edges_by_density(v,d1), seed=111, directed=True))
g2=modded_dict_of_dicts(nx.gnp_random_graph(v,edges_by_density(v,d2), seed=111,directed=True))
g3=modded_dict_of_dicts(nx.gnp_random_graph(v,edges_by_density(v,d3), seed=111,directed=True))

s,d = 1,2

t1 = t.time()
shortest_distance,track_path,predecessors=dijkstra(g1, s,d)
print("Shortest distance is "+str(shortest_distance[d]))
print("Optimal path is "+str(track_path))
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
shortest_distance,track_path,predecessors=dijkstra(g2, s,d)
print("Shortest distance is "+str(shortest_distance[d]))
print("Optimal path is "+str(track_path))
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
shortest_distance,track_path,predecessors=dijkstra(g3, s,d)
print("Shortest distance is "+str(shortest_distance[d]))
print("Optimal path is "+str(track_path))
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
node_data=bellmanford(g1, s,d)
print('Shortest distance: '+str(node_data[d]['cost']))
print('Shortest path: '+str(node_data[d]['pred']+ [d]))
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
node_data=bellmanford(g2, s,d)
print('Shortest distance: '+str(node_data[d]['cost']))
print('Shortest path: '+str(node_data[d]['pred']+ [d]))
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
node_data=bellmanford(g3, s,d)
print('Shortest distance: '+str(node_data[d]['cost']))
print('Shortest path: '+str(node_data[d]['pred']+ [d]))
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
vertices, dist, previous_node = floydwarshall(g1)
reconstruct_path(vertices,previous_node, s,d)
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
vertices, dist, previous_node = floydwarshall(g2)
reconstruct_path(vertices,previous_node, s,d)
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
vertices, dist, previous_node = floydwarshall(g3)
reconstruct_path(vertices,previous_node, s,d)
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
paths=johnson(g1)
print("Shortest distance is "+str(paths[(s,d)]))
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
paths=johnson(g2)
print("Shortest distance is "+str(paths[(s,d)]))
t2 = t.time()
print('serial: ',t2 - t1, 's')

t1 = t.time()
paths=johnson(g3)
print("Shortest distance is "+str(paths[(s,d)]))
t2 = t.time()
print('serial: ',t2 - t1, 's')



