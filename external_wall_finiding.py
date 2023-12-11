import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import deque

import math

def show_graph(edges, node_coord_map):
    # Draw edges
    for edge in edges:
        x0, y0 = node_coord_map[edge[0]]
        x1, y1 = node_coord_map[edge[1]]
        plt.plot([x0, x1], [y0, y1], 'k-', lw=1,  zorder=1)  # 'k-' for black line

    for i in node_coord_map:
        plt.scatter(node_coord_map[i][0], node_coord_map[i][1],  s=700, c='skyblue',  zorder=2)
        plt.text(node_coord_map[i][0], node_coord_map[i][1], str(i), fontsize=12, ha='center', va='center',  zorder=3)
        # plt.text(pos[0], pos[1], str(node), fontsize=12, ha='center', va='center')
    plt.set_title('test graph')


def Draw_layout(loop,node_coord_map):

    edges = [ (loop[n], loop[n+1]) for n in range(len(loop)-1)]
    for edge in edges:
        x0, y0 = node_coord_map[edge[0]]
        x1, y1 = node_coord_map[edge[1]]
        plt.plot([x0, x1], [y0, y1], 'k-', lw=1,  zorder=1)  # 'k-' for black line



def show_graph(edges, node_coord_map):

    # Draw edges
    for edge in edges:
        x0, y0 = node_coord_map[edge[0]]
        x1, y1 = node_coord_map[edge[1]]
        plt.plot([x0, x1], [y0, y1], 'k-', lw=1,  zorder=1)  # 'k-' for black line

    for i in node_coord_map:
        plt.scatter(node_coord_map[i][0], node_coord_map[i][1],  s=700, c='skyblue',  zorder=2)
        plt.text(node_coord_map[i][0], node_coord_map[i][1], str(i), fontsize=12, ha='center', va='center',  zorder=3)
        # plt.text(pos[0], pos[1], str(node), fontsize=12, ha='center', va='center')



def bfs_simple_cycle_v2(neighbors_dict, start):
    visited = set()  # Keep track of visited nodes
    queue = deque([(start, [start])])  # Queue for BFS: (current_node, path_so_far)

    i = 0
    while queue:
        i += 1
        current_node, path = queue.popleft()
        if current_node not in neighbors_dict[start]:
            visited.add(current_node)

        for neighbor in neighbors_dict[current_node]:
            if neighbor == start and len(path) > 2:
                return path + [start]  # Cycle found
            if neighbor not in visited:
                if neighbor not in neighbors_dict[current_node]:
                    visited.add(neighbor)
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))
    return None  # No cycle found





def bfs_simple_cycle(G, start):
    visited = set()  # Keep track of visited nodes
    queue = deque([(start, [start])])  # Queue for BFS: (current_node, path_so_far)

    while queue:
        current_node, path = queue.popleft()
        visited.add(current_node)

        for neighbor in G.neighbors(current_node):
            if neighbor == start and len(path) > 2:
                return path + [start]  # Cycle found
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None  # No cycle found

def find_longest_distance(start, neighbors ,node_coord_map):

    start_coords = node_coord_map[start]
    distances = []
    for n in neighbors:
        current_coords = node_coord_map[n]
        d = ((start_coords[0] - current_coords[0])**2 + (start_coords[1] - current_coords[1])**2)**0.5
        distances.append((d,n))
    distances = sorted(distances, reverse=True)
    sorted_neighbors = [n[1] for n in distances]
    return sorted_neighbors

def compare_cross_product(start, path, neighbors, node_coord_map):

    start_coords = node_coord_map[start]
    # v1 = np.array(node_coord_map[path[-1]]) - np.array(node_coord_map[path[-2]])
    v1 = np.array(node_coord_map[path[-1]]) - np.array(node_coord_map[start])

    Areas = []
    for n in neighbors:

        current_coords = node_coord_map[n]
        v2 = np.array(node_coord_map[n]) - np.array(node_coord_map[path[-1]])
        d = ((start_coords[0] - current_coords[0]) ** 2 + (start_coords[1] - current_coords[1]) ** 2) ** 0.5
        a = abs(v1[0]*v2[1] - v1[1]*v2[0])
        Areas.append((a,d,n))

    Areas = sorted(Areas, reverse=True)
    sorted_neighbors = [n[2] for n in Areas]
    return sorted_neighbors


def angle_between_vectors(v1, v2):
    # Calculate the dot product of the two vectors
    angle1 = np.arctan2(v1[1], v1[0])*180/3.1415
    angle2 = np.arctan2(v2[1], v2[0])*180/3.1415

    return angle2 - angle1


def neighbors_priority(start, path, neighbors, node_coord_map):

    start_coords = node_coord_map[start]
    v1 = np.array(node_coord_map[path[-1]]) - np.array(node_coord_map[path[-2]])
    angle_ref  = round(np.arctan2(v1[1], v1[0]) * 180 / 3.1415 + 180)
    if angle_ref < 0:
        angle_ref = 360 + angle_ref
    if angle_ref >= 360:
        angle_ref = 360 - angle_ref
    # angle_ref = angle + 180
    #v1 = np.array(node_coord_map[path[-1]]) - np.array(node_coord_map[start])

    angles = []
    for n in neighbors:


        current_coords = node_coord_map[n]
        v2 = np.array(node_coord_map[n]) - np.array(node_coord_map[path[-1]])
        angle = round(np.arctan2(v2[1], v2[0]) * 180 / 3.1415)
        if angle < 0:
            angle = 360 + angle
        if angle > 360:
            angle = angle - 360

        alfa = angle_ref - angle
        if alfa < 0:
            alfa = 360 + alfa
        if alfa > 360:
            alfa = alfa - 360


        # alfa =  angle_between_vectors(v1,v2)
        # alfa = np.arcsin(sin_angle)*180/3.1415

        # d = ((start_coords[0] - current_coords[0]) ** 2 + (start_coords[1] - current_coords[1]) ** 2) ** 0.5
        # a = abs(v1[0]*v2[1] - v1[1]*v2[0])
        angles.append((alfa,n))

    # angles = sorted(angles, reverse=True)
    angles = sorted(angles)
    sorted_neighbors = [n[1] for n in angles]
    return sorted_neighbors

def neighbors_priority_start_point(path, neighbors, node_coord_map):
    angles = []
    for n in neighbors:
        v2 = np.array(node_coord_map[n]) - np.array(node_coord_map[path[-1]])
        angle = 180 - round(np.arctan2(v2[1], v2[0]) * 180 / 3.1415)
        if angle < 0:
            angle = 360 + angle
        if angle > 360:
            angle = angle - 360
        angles.append((angle,n))

    angles = sorted(angles)
    sorted_neighbors = [n[1] for n in angles]
    return sorted_neighbors



def dfs(start, node_coord_map, neighbors_dict):
    """
    Perform Depth-First Search (DFS) to find a path from the given start coordinates to the target.
    """
    path = [start]

    # directions = self.directions
    target_coord = start
    visited = np.zeros((len(node_coord_map)), dtype=bool)
    # visited[start] = True

    path_is_finded = False
    current = start
    while (path_is_finded == False):
        dead_end = True
        # for drow, dcol, delev in directions:
        neighbors = neighbors_dict[current]
        if len(path)==1:
            # neighbors = find_longest_distance(start, neighbors, node_coord_map)
            neighbors = neighbors_priority_start_point(path, neighbors, node_coord_map)
        else:
            #neighbors = compare_cross_product(start, path, neighbors, node_coord_map)
            neighbors = neighbors_priority(start, path, neighbors, node_coord_map)

        for neighbor in neighbors:
            if not visited[neighbor]:
                # once we have more that one path in layout the that are visited also are valid
                current = neighbor

                if current == target_coord and len(path) > 2:
                    path_is_finded = True
                    path.append(current)
                    return path
                else:
                    if neighbor != start:
                        visited[neighbor] = True
                        # add the node to the path
                        path.append(current)
                        dead_end = False
                        break
        if dead_end:
            # backtracking
            path.pop()
            current = path[-1]
    # add path to layout
    return path


def dfs_connected_components(graph, start, visited):
    """
    Depth First Search (DFS) algorithm to traverse a graph.
    """
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_connected_components(graph, neighbor, visited)

def find_connected_components_dfs_v1(graph):
    """
    Finds all connected components in the graph using DFS.
    """
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            component = set()
            dfs_connected_components(graph, node, component)
            components.append(component)
            visited.update(component)

    return components

def test_find_connected_coponents():
    # Example graph represented as an adjacency list
    graph = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2],
        4: [5],
        5: [4]
    }

    # Find all connected components
    connected_components = find_connected_components_dfs_v1(graph)
    connected_components


def find_connected_components(start, node_coord_map, neighbors_dict):
    """
    Perform Depth-First Search (DFS) to find a path from the given start coordinates to the target.
    """
    path = [start]

    # directions = self.directions
    target_coord = start
    visited = np.zeros((len(node_coord_map)), dtype=bool)
    # visited[start] = True

    path_is_finded = False
    current = start
    while (path_is_finded == False):
        dead_end = True
        # for drow, dcol, delev in directions:
        neighbors = neighbors_dict[current]
        if len(path)==1:
            neighbors = find_longest_distance(start, neighbors, node_coord_map)
        else:
            #neighbors = compare_cross_product(start, path, neighbors, node_coord_map)
            neighbors = neighbors_priority(start, path, neighbors, node_coord_map)

        for neighbor in neighbors:
            if not visited[neighbor]:
                # once we have more that one path in layout the that are visited also are valid
                current = neighbor

                if current == target_coord and len(path) > 2:
                    path_is_finded = True
                    path.append(current)
                    return path
                else:
                    if neighbor != start:
                        visited[neighbor] = True
                        # add the node to the path
                        path.append(current)
                        dead_end = False
                        break
        if dead_end:
            # backtracking
            path.pop()
            current = path[-1]
    # add path to layout
    return path


def test_case_1():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (20, 10), 4: (10, 10), 5: (0, 10)}
    coord_node_map = {(0, 0): 0,  (10, 0): 1, (20, 0): 2, (20, 10): 3, (10, 10): 4, (0, 10): 5}
    # Add some edges to the graph
    edges = [(0, 2), (2, 3), (3, 5), (5, 0), (1, 4)]

    G = nx.Graph()
    G.add_edges_from(edges)

    # Get a dictionary of neighboring nodes
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}

    circle =dfs(0, node_coord_map, neighbors_dict)

    circle = bfs_simple_cycle_v2(neighbors_dict, 0)
    circle2 = bfs_simple_cycle_v2(neighbors_dict, 2)

    return node_coord_map,coord_node_map, edges


def test_case_2():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (20, 10), 4: (20, 20), 5: (10, 20), 6: (10, 10), 7: (0, 10)}
    coord_node_map = { node_coord_map[key]:key for key in node_coord_map}
    # Add some edges to the graph
    edges = [(0, 1), (1, 4), (4, 5), (5, 0), (1, 2), (2, 3), (3, 4)]

    G = nx.Graph()
    G.add_edges_from(edges)

    # Get a dictionary of neighboring nodes
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}

    circle = dfs(0, node_coord_map, neighbors_dict)
    circle = bfs_simple_cycle_v2(neighbors_dict, 0)
    circle2 = bfs_simple_cycle_v2(neighbors_dict, 2)
    return


def test_case_3():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (20, 10), 4: (20, 20), 5: (10, 20), 6: (10, 10), 7: (0, 10)}
    coord_node_map = { node_coord_map[key]:key for key in node_coord_map}
    # Add some edges to the graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (1, 6), (3, 6)]

    G = nx.Graph()
    G.add_edges_from(edges)

    # Get a dictionary of neighboring nodes
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}

    circle = dfs(0, node_coord_map, neighbors_dict)
    circle = bfs_simple_cycle_v2(neighbors_dict, 0)
    circle2 = bfs_simple_cycle_v2(neighbors_dict, 2)
    return


def test_case_4():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (30, 0), 3: (30, 10), 4: (30, 20), 5: (10, 20), 6: (10, 10), 7: (0, 10)}
    coord_node_map = { node_coord_map[key]:key for key in node_coord_map}
    # Add some edges to the graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (1, 6), (3, 6)]

    show_graph(edges, node_coord_map)

    G = nx.Graph()
    G.add_edges_from(edges)

    # Get a dictionary of neighboring nodes
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}

    circle = dfs(0, node_coord_map, neighbors_dict)
    circle = bfs_simple_cycle_v2(neighbors_dict, 0)
    circle2 = bfs_simple_cycle_v2(neighbors_dict, 2)
    return



def test_case_5():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (30, 0), 3: (30, 10), 4: (30, 20), 5: (10, 20), 6: (10, 10), 7: (0, 10),
                      8: (20,0), 9: (20,10), 10: (20,20)}
    coord_node_map = { node_coord_map[key]:key for key in node_coord_map}
    # Add some edges to the graph
    edges = [(0, 1), (1, 8),  (8, 2), (2, 3),  (3, 4), (4, 10), (10, 5), (5, 6), (6, 7), (7, 0), (1, 6), (3, 9), (9, 6),
             (9,10), (9, 10), (8, 9)]

    show_graph(edges, node_coord_map)

    G = nx.Graph()
    G.add_edges_from(edges)
    fig = plt.figure()
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}
    circle = dfs(0, node_coord_map, neighbors_dict)
    Draw_layout(circle,node_coord_map )

    return



def test_case_6():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (30, 0), 3: (30, 10), 4: (30, 20), 5: (10, 20), 6: (10, 10), 7: (0, 10),
                      8: (20,0), 9: (20,10), 10: (20,20), 11: (0,20), 12: (0,30), 13: (30,30)}
    coord_node_map = { node_coord_map[key]:key for key in node_coord_map}
    # Add some edges to the graph
    edges = [(0, 1), (1, 8),  (8, 2), (2, 3),  (3, 4), (4, 10), (10, 5), (5, 6), (6, 7), (7, 0), (1, 6), (3, 9), (9, 6),
             (9,10), (9, 10), (8, 9), (5,11), (11,12), (12,13), (13,4)]

    show_graph(edges, node_coord_map)

    G = nx.Graph()
    G.add_edges_from(edges)
    fig = plt.figure()
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}
    circle = dfs(0, node_coord_map, neighbors_dict)
    Draw_layout(circle,node_coord_map )


def test_case_7():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (30, 0), 3: (30, 10), 4: (30, 20), 5: (10, 20), 6: (10, 10), 7: (0, 10),
                      8: (20,0), 9: (20,10), 10: (20,20), 11: (0,20), 12: (0,30), 13: (30,30), 14: (40,20), 15: (40,10)}
    coord_node_map = { node_coord_map[key]:key for key in node_coord_map}
    # Add some edges to the graph
    edges = [(0, 1), (1, 8),  (8, 2), (2, 3),  (3, 4), (4, 10), (10, 5), (5, 6), (6, 7), (7, 0), (1, 6), (3, 9), (9, 6),
             (9,10), (9, 10), (8, 9), (5,11), (11,12), (12,13), (13,4), (4,14), (14,15), (15,3)]

    show_graph(edges, node_coord_map)

    G = nx.Graph()
    G.add_edges_from(edges)
    fig = plt.figure()
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}
    circle = dfs(0, node_coord_map, neighbors_dict)
    Draw_layout(circle,node_coord_map )

def find_initial_corner(comp, node_coord_map):

    nodes = []
    for n in comp:
        nodes.append((node_coord_map[1],node_coord_map[0], n ))

    sorted_nodes = sorted(nodes)
    return sorted_nodes[0][2]



def test_case_8():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (20, 10), 4: (10, 10), 5: (0, 10),
                      6: (40, 0), 7: (50, 0), 8: (60, 0), 9: (60, 10), 10: (50, 10), 11: (40, 10),}

    coord_node_map = {node_coord_map[key]: key for key in node_coord_map}
    # Add some edges to the graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4,   5), (5, 0), (1, 4),
             (6, 7), (7, 8), (8, 9),(9 , 10), (10, 11), (11, 6), (7,10)]




    show_graph(edges, node_coord_map)

    G = nx.Graph()
    G.add_edges_from(edges)
    fig = plt.figure()
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}

    connected_components = find_connected_components_dfs_v1(neighbors_dict)

    for comp in connected_components:
        start = find_initial_corner(comp, node_coord_map)
        circle = dfs(start, node_coord_map, neighbors_dict)
        Draw_layout(circle, node_coord_map)


def test_case_9():
    # Manually define the position of each node
    node_coord_map = {0: (0, 0), 1: (10, 0), 2: (30, 0), 3: (30, 10), 4: (30, 20), 5: (10, 20), 6: (10, 10), 7: (0, 10),
                      8: (20, 0), 9: (20,10), 10: (20,20), 11: (0,20), 12: (0,30), 13: (30,30), 14: (40,20), 15: (40,10),
                      16: (50, 0), 17: (60, 0), 18:(60,10), 19: (50,10) }

    # Add some edges to the graph
    edges = [(0, 1), (1, 8),  (8, 2), (2, 3),  (3, 4), (4, 10), (10, 5), (5, 6), (6, 7), (7, 0), (1, 6), (3, 9), (9, 6),
             (9,10), (9, 10), (8, 9), (5,11), (11,12), (12,13), (13,4), (4,14), (14,15), (15,3), (16,17), (17,18),
             (18,19),(19,16)]

    show_graph(edges, node_coord_map)

    G = nx.Graph()
    G.add_edges_from(edges)
    fig = plt.figure()
    neighbors_dict = {n: list(neighbors) for n, neighbors in G.adjacency()}

    connected_components = find_connected_components_dfs_v1(neighbors_dict)

    for comp in connected_components:
        start = find_initial_corner(comp, node_coord_map)
        circle = dfs(start, node_coord_map, neighbors_dict)
        Draw_layout(circle, node_coord_map)




def draw_graph(node_coord_map, G):
    # Draw the graph with specified positions
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos=node_coord_map, with_labels=True, font_weight='bold', node_color='skyblue', node_size=700)
    plt.title("Graph Visualization with Manual Coordinates")
    plt.show()

def main():
    # test_find_connected_coponents()
    # node_coord_map,coord_node_map, edges = test_case_1()
    # node_coord_map, coord_node_map, edges = test_case_2()
    # node_coord_map, coord_node_map, edges = test_case_3()
    # node_coord_map, coord_node_map, edges = test_case_4()
    # test_case_5()
    # test_case_6()
    test_case_7()
    test_case_8()
    test_case_9()

if __name__ == "__main__":
    main()