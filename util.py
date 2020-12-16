import networkx as nx
import csv
import numpy as np
import random

def remove_edges(G, miss_edges):
    '''
    '''
    G_miss = G.copy()
    edges = list(G.edges())
    to_remove = random.sample(edges, miss_edges)

    for e in to_remove:
        G_miss.remove_edge(e[0],e[1])

    return G_miss

def read_contact_net(input_file_name, threshold, sep=' '):
    '''
    	Reads contact network from input file and returns 
	a networkx graph.

	input file format:
		timestamp i j

	threshold is a minimum duration for a contact (in sec.)

    '''
    G = nx.Graph()
    idx = {}

    with open(input_file_name, 'r') as file_in:
        reader = csv.reader(file_in, delimiter = sep)
        contacts = {}

        for r in reader:
            u = r[1]
            v = r[2]

            time = int(r[0])

            if u > v:
                v = r[1]
                u = r[2]

            if (u,v) not in contacts:
                contacts[(u,v)] = [time]
            else:
                contacts[(u,v)].append(time)

            if u not in idx:
                idx[u] = len(idx)

            if v not in idx:
                idx[v] = len(idx)

    for v in idx:
        G.add_node(idx[v])

    cont_contacts = {}

    for e in contacts:
        cont_contacts[e] = []

        cnt = [contacts[e][0], contacts[e][0]]

        for i in range(1, len(contacts[e])):
            if contacts[e][i] == (cnt[-1] + 20):
                cnt[-1] = contacts[e][i]
            else:
                cont_contacts[e].append(cnt)
                cnt = [contacts[e][i], contacts[e][i]]

        cont_contacts[e].append(cnt)

    edges = {}
    for (u,v) in cont_contacts:
        for c in cont_contacts[u,v]:
            dur = (c[1]-c[0])

            if dur > threshold:
                if (u,v) not in edges:
                    edges[(u,v)] = dur
                else:
                    edges[(u,v)] = edges[(u,v)] + dur

    weight_max = np.max(list(edges.values()))
    for (u,v) in edges:
        G.add_edge(idx[u],idx[v], weight=edges[(u,v)]/weight_max)

    return G

def read_clusters(input_file_name, G, sep):
    '''
        Reads clusters from file and returns lists of lists 
	with clusters and their members for nodes in G.

	input file format:
		node cluster_id

    '''
    with open(input_file_name, 'r') as file_in:
        reader = csv.reader(file_in, delimiter = sep)
        clusters = {}

        for r in reader:
            node = r[0]
            cluster = r[1]
            if node in G.nodes():
                if cluster not in clusters:
                    clusters[cluster] = []

                clusters[cluster].append(node)

    clusters_list = []

    for c in clusters:
        clusters_list.append(clusters[c])

    return clusters_list
