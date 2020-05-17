import networkx as nx
from trajecten import get_paths
from trajecten import station_data


import matplotlib.pyplot as plt

def get_coords(node, table):
    coord = [[t[2], t[1]] for t in table if t[0] == node]
    if len(coord) != 0:
        return [float(coord[0][0]), float(coord[0][1])]
    else:
        return []

def get_coords2(node, coords, edges):
    connected_nodes = []
    for e in edges:
        if node == e[0]:
            if e[1] not in connected_nodes:
                connected_nodes.append(e[1])
        elif node == e[1]:
            if e[0] not in connected_nodes:
                connected_nodes.append(e[0])
    connected_coords = [get_coords(node, coords) for node in connected_nodes if get_coords(node, coords) != []]

    x = sum([c[0] for c in connected_coords]) / len(connected_coords)
    y = sum([c[1] for c in connected_coords]) / len(connected_coords)
    return [x, y]

def create_graph():
    paths = [p[0] for p in get_paths("data/ns.csv")]
    stations = station_data()

    # Get all edges and nodes of the graph
    nodes = []
    edges = {}
    for p in paths:
        for i in range(len(p)):
            # Append to nodes
            current_node = p[i]
            if current_node not in nodes:
                nodes.append(current_node)
            # Append to edges
            if i != 0:
                previous_node = p[i - 1]
                if (previous_node, current_node) not in edges.keys():
                    edges[(previous_node, current_node)] = 1
                else:
                    edges[(previous_node, current_node)] = edges[(previous_node, current_node)] + 1

    # Get node coordinates
    node_coords = []
    for node in nodes:
        coords = [[s[1], s[2], s[3]] for s in stations]
        # get coordinates using the coords table
        coord = get_coords(node, coords)
        # if coordinates couldn't be found, get coordinates using the connected
        # nodes
        if coord == []:
            coord = get_coords2(node, coords, edges)
        x, y = coord
        node_coords.append([[x, y], node])

    G = nx.Graph()
    #Add nodes to network with specific coordinates
    for i in range(len(nodes)):
        coord = tuple(node_coords[i][0])
        name = node_coords[i][1]
        G.add_node(name, pos=coord)
    #get position attributes from network
    pos=nx.get_node_attributes(G,'pos')

    weighted_edges = []
    #add weighted edges
    for edge_key in edges.keys():
        #convert edge to tuple, (x, y, weight)
        weight = edges[edge_key]
        edge = edge_key + ({'weight': weight**2},)
        weighted_edges.append(edge)

    G.add_weighted_edges_from(weighted_edges)

    img = plt.imread("download.gif")
    fig, ax = plt.subplots()
    #trial and error for correct extent fit
    ax.imshow(img, extent=[3.35,7.15,50.7,53.5])

    nx.draw(G,pos,node_size=10, with_labels=True, font_size = 7)
    plt.show()

create_graph()
