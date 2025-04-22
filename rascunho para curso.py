"""
def tower_of_hanoi(count, stacks=None, source=0, auxiliary=1, destination=2, moves=0):
    if not stacks:
        stacks = [['ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in range(count-1, -1, -1)], [], []]
        moves = 1
        print(stacks)
    
    if(count==1):
        if(stacks[source]):
            stacks[destination].append(stacks[source].pop())
            print(stacks)
    else:
        tower_of_hanoi(count-1,stacks,source,destination,auxiliary,moves+1)
        stacks[destination].append(stacks[source].pop())
        print(stacks)
        tower_of_hanoi(count-1,stacks,auxiliary,source,destination,moves+1)
        


tower_of_hanoi(3)
"""
"""
def DFS(map, coord=(0,0)):
    coordX=coord[0]
    coordY=coord[1]
    if(coordY>=len(map) or coordX>=len(map[0]) or coordY<0 or coordX<0):
        return 0
    
    if(map[coordY][coordX]==1):
        map[coordY][coordX]=2
        DFS(map,(coordX-1,coordY-1))
        DFS(map,(coordX,coordY-1))
        DFS(map,(coordX+1,coordY-1))
        DFS(map,(coordX-1,coordY))
        DFS(map,(coordX+1,coordY))
        DFS(map,(coordX-1,coordY+1))
        DFS(map,(coordX,coordY+1))
        DFS(map,(coordX+1,coordY+1))
        return 1
    return 0

def get_groups(map):
    grupos=0
    for i in range(len(map)): #coluna
        for j in range(len(map[0])): #linha
            grupos+=DFS(map,coord=(j,i))
    return grupos

	
map = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

print(get_groups(map))

"""
class Vertex:
    def __init__(self, value):
        self._value = value
    def __repr__(self):
        return f'<Vertex: {self._value}>'
    def __hash__(self):
        return hash(id(self))


class Edge:
    def __init__(self, u, v, x):
        self._first = u
        self._second = v
        self._value = x
    def __repr__(self):
        return f'<Edge ({self._value}): {self._first} --> {self._second}>'
    def endpoints(self):
        return (self._first, self._second)
    def opposite(self, v):
        return self._second if v is self._first else self._first
    def value(self):
        return self._value
    def __hash__(self):
        return hash( (self._first, self._second) )


class Graph:
    def __init__(self, adj_map = None):
        if adj_map:
            self._adj_map = adj_map
        else:
            self._adj_map = {}
    def get_vertices(self):
        return self._adj_map.keys()
    def get_edges(self):
        """Return a set of all edges of the graph."""
        result = set()
        for secondary_map in self._adj_map.values():
            result.update(secondary_map.values())
        return result
    def get_edge(self, u, v):
        """
        Returns the edge from u to v, or None if not adjacents.
        """
        return self._adj_map[u].get(v)
    def degree(self, u):
        """
        Returns the number of edges incident to vertex u
        """
        return len(self._adj_map[u])
    def get_adjacent_vertices(self, u):
        """
        Return a list of the adjacent vertices of a given vertex
        """
        return list(self._adj_map[u].keys())
    def get_incident_edges(self, u):
        """
        Returns edges incident to vertex u
        """
        return list(self._adj_map[u].values())
    def add_vertex(self, value):
        vertex = Vertex(value)
        self._adj_map[vertex] = {}
        return vertex
    def add_edge(self, u, v, x=None):
        edge = Edge(u, v, x)
        self._adj_map[u][v] = edge
        self._adj_map[v][u] = edge
    def get_adj_map(self):
        return self._adj_map
    def get_adj_matrix(self):
        all_vertices = self._adj_map.keys()
        return [[int(bool(self._adj_map[u].get(v))) for v in all_vertices] for u in all_vertices]


def dijkstra_shortest_path(source_vertex, graph):
    dsp={}
    notVisited=[]
    for v in graph.get_vertices():
        dsp[v]={'shortest':float('inf'), 'previous':None}
        notVisited.append(v)

    dsp[source_vertex]['shortest']=0
    
    while(len(notVisited)>0):
        current=findTheShortest(dsp,notVisited)
        for v in graph.get_adjacent_vertices(current):
            somaateAqui=dsp[current]['shortest']+graph.get_edge(current, v).value()
            if(somaateAqui<dsp[v]['shortest']):
                dsp[v]['shortest']=somaateAqui
                dsp[v]['previous']=current
                
        notVisited.remove(current)
        
    
    return dsp

def findTheShortest(dsp,notVisited):
    shortest=None
    for key in notVisited:
        if(not shortest):
            shortest=key
        if(dsp[key]['shortest']<dsp[shortest]['shortest']):
            shortest=key
    return shortest


def get_best_city(graph):
    totais=(None,153454512)
    for i in graph.get_vertices():
        dps=dijkstra_shortest_path(i, graph)
        print(i)
        total=0
        for key in dps:
            print("   ",key,"custo:",dps[key]['shortest'])
            total+=dps[key]['shortest']
        if total<totais[1]:
            totais=(i._value,total)
        print("total=",total)
    return totais


A = Vertex('A')
B = Vertex('B')
C = Vertex('C')
D = Vertex('D')
E = Vertex('E')
F = Vertex('F')
G = Vertex('G')
H = Vertex('H')
J = Vertex('J')

AB = Edge(A, B, 25)
AD = Edge(A, D, 3)
AG = Edge(A, G, 17)
BC = Edge(B, C, 2)
BD = Edge(B, D, 73)
BE = Edge(B, E, 84)
CF = Edge(C, F, 79)
DE = Edge(D, E, 47)
DH = Edge(D, H, 10)
EF = Edge(E, F, 73)
EH = Edge(E, H, 15)
FJ = Edge(F, J, 48)
GH = Edge(G, H, 38)
HJ = Edge(H, J, 72)

am = {
    A: {B: AB, D: AD, G: AG},
    B: {A: AB, C: BC, D: BD, E: BE},
    C: {B: BC, F: CF},
    D: {A: AD, B: BD, E: DE, H: DH},
    E: {B: BE, D: DE, F: EF, H: EH},
    F: {C: CF, E: EF, J: FJ},
    G: {A: AG, H: GH},
    H: {D: DH, E: EH, G: GH, J: HJ},
    J: {F: FJ, H: HJ}
}

g = Graph(am)

print(get_best_city(g))