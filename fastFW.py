import sys
from multiprocessing.pool import ThreadPool
from functools import partial
import osmnx
import time as t
import random
import networkx as nx

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

def InnerLoop(i, dist,previous_node, n, k):
    for j in range(n):
        if dist[i][j]>dist[i][k]+ dist[k][j]:
            dist[i][j] = dist[i][k] + dist[k][j]
            previous_node[i][j] = previous_node[k][j]
    return (i,dist[i],previous_node[i])


def floydwarshall2(graph):
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
    print(n)
    # Initialize predecessor (previous_node) matrix
    previous_node = [[None] * n for _ in range(n)]
    # Initialize previous_node for direct edges
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] != inf:
                previous_node[i][j] = vertices[i]  # Previous node is the source node
    # Main Floyd-Warshall algorithm   

    pool = ThreadPool(14)
    for k in range(n):
        p = partial(InnerLoop, dist=dist,previous_node=previous_node,n=n,k=k)
        result_list=pool.map(p,range(n))
        for element in result_list:
            dist[element[0]]=element[1]
            previous_node[element[0]]=element[2]
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

if __name__ == '__main__':

    G = osmnx.graph_from_point((50.3869524, 30.4807769), dist=920)

    s =  960023629
    d =  334465761 
    
    source='A'
    destination='F'

    graph2={
    'A':{'B':6,'C':4,'D':5},
    'B':{'E':-1},
    'C':{'B':-2, 'E':3},
    'D':{'C':-2,'F':-1},
    'E':{'F':3},
    'F':{}
    }
    '''
    t1 = t.time()
    vertices, dist, previous_node = floydwarshall2(graph_from_osmnx(G))
    reconstruct_path(vertices,previous_node, s, d)
    t2 = t.time()
    print('serial: ',t2 - t1, 's')
    '''
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

    print("Parallel Floyd-Warshall algorithm:")
    t1 = t.time()
    vertices, dist, previous_node = floydwarshall2(graph2)
    reconstruct_path(vertices,previous_node, source, destination)
    t2 = t.time()
    print('serial: ',t2 - t1, 's')

    t1 = t.time()
    vertices, dist, previous_node = floydwarshall2(g1)
    reconstruct_path(vertices,previous_node, s, d)
    t2 = t.time()
    print('serial: ',t2 - t1, 's')

    t1 = t.time()
    vertices, dist, previous_node = floydwarshall2(g2)
    reconstruct_path(vertices,previous_node, s, d)
    t2 = t.time()
    print('serial: ',t2 - t1, 's')
    
    t1 = t.time()
    vertices, dist, previous_node = floydwarshall2(g3)
    reconstruct_path(vertices,previous_node, s, d)
    t2 = t.time()
    print('serial: ',t2 - t1, 's')
