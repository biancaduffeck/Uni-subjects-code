"""Function to find the groups in a matrix based graph using recursive depth first search."""
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
