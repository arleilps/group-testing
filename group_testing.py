import networkx as nx
import csv
import numpy as np
import scipy
import random

def best_case(full_data, size, time):
    '''
    '''
    statuses = full_data.get_statuses(time=time)
    n_positive = 0
    for v in statuses:
        if statuses[v] in ['I', 'R']:
            n_positive = n_positive + 1
                
    return np.ceil(len(statuses) / size) + size * np.ceil(n_positive/size)

def worst_case(full_data, size, time):
    '''
    '''
    statuses = full_data.get_statuses(time=time)
    n_positive = 0
    for v in statuses:
        if statuses[v] in ['I', 'R']:
            n_positive = n_positive + 1
    
    return np.ceil(len(statuses) / size) + min(n_positive * size, len(statuses))
                
def group_test(full_data, group, time):
    '''
        Tests the group with 100% accuracy
    '''
    statuses = full_data.get_statuses(time=time)
    
    for v in statuses:
        if v in group:
            if statuses[v] in ['I', 'R']:
                return True
    
    return False

def random_groups(G, size):
    '''
        Generates random groups for testing.
    '''
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    
    groups = [[]]
    for i in range(len(nodes)):
        if len(groups[-1]) == size:
            groups.append([])
            
        groups[-1].append(nodes[i])
    
    return groups

def cluster_groups(clusters, size):
    '''
    '''
    clusters.sort(key=len, reverse=True)
    
    clustered_nodes = []
    for c in range(len(clusters)):
        clustered_nodes = clustered_nodes + clusters[c]
    
    groups = [[]]
    for i in range(len(clustered_nodes)):
        if len(groups[-1]) == size:
            groups.append([])
            
        groups[-1].append(clustered_nodes[i])
        
    return groups


def evaluate_two_level_group_testing(infec_matrix, groups):
    '''
    '''
    n_tests = len(groups) * np.ones(infec_matrix.shape[1])
    total = 0
    
    for g in range(len(groups)):
        sz = len(groups[g])
        total = total+sz
        
        group = []
        for v in groups[g]:
            v = int(v)
            if v < infec_matrix.shape[0]:
                group.append(v)
        
        ind_test = infec_matrix[group].sum(axis=0)
        ind_test[ind_test > 0] = 1
        n_tests = n_tests + sz * ind_test
        
    return n_tests / total
