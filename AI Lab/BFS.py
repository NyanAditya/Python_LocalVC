from collections import deque

class Graph:
    def __init__(self, directed=False):
        self.graph = {}  # Dictionary to store the adjacency list
        self.directed = directed
        
    def add_edge(self, u, v):
        # Add edge from u to v
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
        
        # If graph is undirected, add edge from v to u as well
        if not self.directed:
            if v not in self.graph:
                self.graph[v] = []
            self.graph[v].append(u)
    
    def bfs(self, start_vertex):
        """
        Perform Breadth-First Search starting from start_vertex
        """
        visited = set()  # Set to keep track of visited vertices
        queue = deque([start_vertex])  # Queue for BFS
        visited.add(start_vertex)
        
        traversal = []  # List to store BFS traversal order
        
        while queue:
            # Dequeue a vertex from queue
            vertex = queue.popleft()
            traversal.append(vertex)
            
            # Get all adjacent vertices
            # If an adjacent vertex is not visited, mark it visited and enqueue it
            for neighbor in self.graph.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return traversal
    
    def shortest_path(self, start_vertex, end_vertex):
        """
        Find shortest path from start_vertex to end_vertex using BFS
        """
        if start_vertex == end_vertex:
            return [start_vertex]
            
        visited = set([start_vertex])
        queue = deque([start_vertex])
        parent = {}  # To reconstruct the path
        
        while queue:
            vertex = queue.popleft()
            
            for neighbor in self.graph.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = vertex
                    queue.append(neighbor)
                    
                    if neighbor == end_vertex:
                        # Reconstruct the path
                        path = [end_vertex]
                        while path[-1] != start_vertex:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
        
        return None  # No path exists

if __name__ == "__main__":
    # Create a graph
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)
    
    print("BFS traversal starting from vertex 2:")
    print(g.bfs(2))
    
    print("\nShortest path from 0 to 3:")
    path = g.shortest_path(0, 3)
    if path:
        print(" -> ".join(map(str, path)))
    else:
        print("No path exists")