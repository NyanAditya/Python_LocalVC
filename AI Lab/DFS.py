def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = []
    if start not in visited:
        visited.append(start)
        for neighbor in graph[start]:
            dfs_recursive(graph, neighbor, visited)
    return visited

def dfs_iterative(graph, start):
    visited = []
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            stack.extend(reversed([n for n in graph[vertex] if n not in visited]))
    
    return visited

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print(dfs_recursive(graph, 'A'))
print(dfs_iterative(graph, 'A'))