import networkx as nx
import csv
import numpy as np
import scipy
import random
import EoN
from scipy.sparse import csr_matrix

def run_epidemics(G, tau, gamma, weight, num_init, perc_infec, full=False):
    '''
    '''
    time = None
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    n_iter = 0
    
    while time is None:
        np.random.shuffle(nodes)

        full_data = EoN.fast_SIR(G, tau, gamma, initial_infecteds=nodes[0:num_init], 
        transmission_weight=weight, return_full_data=True)

        infec = full_data.I()
        recov = full_data.R()

        for i in range(infec.shape[0]):
            if infec[i] + recov[i] >= perc_infec * n_nodes:
                time = full_data.t()[i]
                break
        
        n_iter += 1
        
        if n_iter > 1000:
            print("Could not reach desired infection rate after 1000 iterations")
        
            return None
    
    if full is True:
        return full_data
    else:
        return full_data.get_statuses(time=time)

def _sample_epidemics(G, tau, gamma, n_samples, perc_infec, num_init, weight='weight'):
    '''
    '''
    n_nodes = G.number_of_nodes()
    nodes = []
    samples = []
    infected = []
    node_index = {}
    
    for v in G.nodes():
        node_index[v] = len(node_index)
    
    for i in range(n_samples):                
        statuses = run_epidemics(G, tau, gamma, weight, num_init, perc_infec)
        
        for v in statuses:
            if statuses[v] in ['I', 'R']:
                node_id = node_index[v]
                
                nodes.append(node_id)
                samples.append(i)
                infected.append(1.)
        
    infection_matrix = csr_matrix((infected, (nodes, samples)), shape=(n_nodes, n_samples))
    
    return infection_matrix, node_index

def sample_epidemics(G, tau, gamma, n_samples, perc_infec, num_init, weight='weight'):
    '''
    '''
    n_nodes = G.number_of_nodes()
    nodes = []
    samples = []
    infected = []
    
    for i in range(n_samples):                
        statuses = run_epidemics(G, tau, gamma, weight, num_init, perc_infec)
        
        for v in statuses:
            if statuses[v] in ['I', 'R']:
                nodes.append(v)
                samples.append(i)
                infected.append(1.)
    
    infection_matrix = csr_matrix((infected, (nodes, samples)), shape=(n_nodes, n_samples))
    
    return infection_matrix

def infection_lists(infection_matrix):
    infec_lists = []
    row, col = infection_matrix.nonzero()
    
    for v in range(infection_matrix.shape[0]):
        infec_lists.append([])
    
    for i in range(row.shape[0]):
        v = row[i]
        s = col[i]
        
        infec_lists[v].append((s,1))
    
    for v in range(infection_matrix.shape[0]):
        infec_lists[v].sort()
        
    return infec_lists
