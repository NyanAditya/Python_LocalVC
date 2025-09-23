from collections import deque

class Graph:
    def __init__(self, directed=False):
        self.graph = {}
        self.directed = directed
        
    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
        
        if not self.directed:
            if v not in self.graph:
                self.graph[v] = []
            self.graph[v].append(u)
    
    def bfs(self, start_vertex):
        visited = set()
        queue = deque([start_vertex])
        visited.add(start_vertex)
        
        traversal = []
        
        while queue:
            vertex = queue.popleft()
            traversal.append(vertex)
            
            for neighbor in self.graph.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return traversal

if __name__ == "__main__":
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)
    
    print("BFS traversal starting from vertex 2:")
    print(g.bfs(2))
