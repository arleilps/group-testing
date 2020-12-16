import numpy as np
import scipy
import matplotlib.pyplot as plt
import EoN
import random
import community as community_louvain
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
from itertools import count
from networkx.utils import not_implemented_for, py_random_state, BinaryHeap
from networkx.algorithms.community.community_utils import is_partition
import heapq

from group_testing import *
from epidemics import *

def modularity_min(G, weight='weight'):
    '''
        Modularity minimization clustering.
        Returns list of clusters.
    '''
    if weight is None:
        vertex_cluster = community_louvain.best_partition(G)
    else:
        vertex_cluster = community_louvain.best_partition(G, weight=weight)
            
    clusters = []
        
    for c in range(np.max(list(vertex_cluster.values()))+1):
        clusters.append([])
    
    for v in vertex_cluster:
        c = vertex_cluster[v]
        clusters[c].append(v)
    
    return clusters

def spectral_clustering(G, k, weight='weight'):
    '''
        Spectral clustering with k clusters.
        Returns list of clusters
    ''' 
    adj_mat = nx.to_numpy_matrix(G, weight=weight)
    sc = SpectralClustering(k, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)
    n_clusters = int(np.max(sc.labels_))+1
    
    clusters = []
    for c in range(n_clusters):
        clusters.append([])
    
    i = 0
    for v in G.nodes():
        c = sc.labels_[i]
        clusters[c].append(v)
        
        i = i + 1
        
    return clusters

def topol_greedy_group(G, group_size, weight='weight'):
    '''
        Greedily groups two sets of nodes by maximizing 
        the total weight inside groups at each step.
    '''
    group_assign = {}
    groups = []
    
    #Groups initialized as singletons
    for v in G.nodes():
        group_assign[v] = len(groups)
        groups.append([v])

    #Each edge is a candidate merge
    #weights stores cross group weights for pairs
    weights = {}
    heap = []
    for e in G.edges:
        if weight is None:
            w = 1.
        else:
            w = G.edges[e]['weight']
        
        idx_i = min(group_assign[e[0]], group_assign[e[1]])
        idx_j = max(group_assign[e[0]], group_assign[e[1]])
        
        if idx_i not in weights:
            weights[idx_i] = {}
            
        if idx_j not in weights:
            weights[idx_j] = {}
        
        weights[idx_i][idx_j] = w
        weights[idx_j][idx_i] = w
        
        heap.append((-w, (idx_i,idx_j)))
    
    heapq.heapify(heap)
    
    while len(heap) > 0:
        item = heapq.heappop(heap)
        idx_i = item[1][0]
        idx_j = item[1][1]

        if len(groups[idx_i]) > 0 and len(groups[idx_j]) > 0:
            if len(groups[idx_i])+len(groups[idx_j]) <= group_size:
                
		#Picks next merge and creates new group
                idx_new = len(groups)
                weights[idx_new] = {}
                groups.append(groups[idx_i]+groups[idx_j])
                
                groups[idx_i] = []
                groups[idx_j] = []
                
		#Updates weights for new group
                for c in weights[idx_i]:
                    if c != idx_j and len(groups[c]) > 0:
                        w = weights[idx_i][c]
                        
                        if c in weights[idx_j]:
                            w = w + weights[idx_j][c]
                        
                        weights[idx_new][c] = w
                        weights[c][idx_new] = w
                        
                        heapq.heappush(heap, (-w, (c, idx_new)))
                        
                for c in weights[idx_j]:
                    if c != idx_i and len(groups[c]) > 0 and c not in weights[idx_new]:
                        w = weights[idx_j][c]
                        
                        if c in weights[idx_i]:
                            w = w + weights[idx_i][c]
                            
                        weights[idx_new][c] = w
                        weights[c][idx_new] = w
                        
                        heapq.heappush(heap, (-w, (c, idx_new)))
                        
    non_empty_groups = []   
    for g in groups:
        if len(g) > 0:
            non_empty_groups.append(g)
            
    return cluster_groups(non_empty_groups, group_size)

def union_sorted(u_inf, v_inf):
    '''
        Returns the union of u_inf and v_inf, which
	are lists of infection events
	that are assumed to be sorted.
    '''
    union = []
    
    i = 0
    j = 0
    
    while i < len(u_inf) and j < len(v_inf):
        if u_inf[i][0] < v_inf[j][0]:
            if u_inf[i][1] > 0:
                union.append(u_inf[i])
            i = i + 1
        elif v_inf[j][0] < u_inf[i][0]:
            if v_inf[j][1] > 0:
                union.append(v_inf[j])
            j = j + 1
            
        else:
            if u_inf[i][1]+v_inf[j][1] > 0:
                union.append((u_inf[i][0],u_inf[i][1]+v_inf[j][1]))
            i = i + 1
            j = j + 1
            
    while i < len(u_inf):
        if u_inf[i][1] > 0:
            union.append(u_inf[i])
        i = i + 1
        
    while j < len(v_inf):
        if v_inf[j][1] > 0:
            union.append(v_inf[j])
        j = j + 1
        
    return union

def diff_sorted(group_inf, v_inf):
    '''
        Returns the difference between 
	group_inf and v_inf, which
	are lists of infection events
	that are assumed to be sorted.
    '''
    union = []
    diff = []
    
    i = 0
    j = 0
    
    while i < len(group_inf) and j < len(v_inf):
        if group_inf[i][0] < v_inf[j][0]:
            if group_inf[i][1] > 0:
                diff.append(group_inf[i])
            i = i + 1
        elif v_inf[j][0] < group_inf[i][0]:
            j = j + 1
            
        else:
            if group_inf[i][1] > 1:
                diff.append((group_inf[i][0],group_inf[i][1]-1))
            i = i + 1
            j = j + 1
            
    while i < len(group_inf):
        if group_inf[i][1] > 0:
            diff.append(group_inf[i])
        i = i + 1
    
    return diff
            
def samp_greedy_group(G, group_size, infec_matrix):
    '''
        Greedily groups two sets of nodes by minimizing 
        the expected number of tests.
    '''
    group_assign = {}
    groups = []
    n_samples = infec_matrix.shape[1]
    group_infecs = []
    score = G.number_of_nodes() * (1. + infec_matrix.sum() / (n_samples * G.number_of_nodes()))
    
    infec_lists = infection_lists(infec_matrix)
    
    #Each node as a group
    for v in G.nodes():
        group_assign[int(v)] = len(groups)
        groups.append([int(v)])
        group_infecs.append(infec_lists[int(v)])
    
    #Initializing heap
    heap = []
    for e in G.edges():
        u = int(e[0])
        v = int(e[1])
        
        u_inf = infec_lists[u]
        v_inf = infec_lists[v]
        uv_inf = union_sorted(u_inf, v_inf)

        w = 1. - 2. * (len(uv_inf)/n_samples) + len(u_inf) / n_samples + len(v_inf) / n_samples

        heap.append((-w, (group_assign[u],group_assign[v])))
    
    heapq.heapify(heap)
    
    while len(heap) > 0:
        item = heapq.heappop(heap)
        idx_i = item[1][0]
        idx_j = item[1][1]
        w = item[0]
        
        if len(groups[idx_i]) > 0 and len(groups[idx_j]) > 0:
            score = score + w
            if len(groups[idx_i])+len(groups[idx_j]) <= group_size:
                idx_new = len(groups)
                groups.append(groups[idx_i]+groups[idx_j])
                group_infecs.append(union_sorted(group_infecs[idx_i],group_infecs[idx_j])) 
                
                groups[idx_i] = []
                groups[idx_j] = []
                group_infecs[idx_i] = None
                group_infecs[idx_j] = None
                
                new_inf = group_infecs[-1]
                
                for c in range(len(groups)-1):
                    if len(groups[c]) > 0:
                        c_inf = group_infecs[c]       
                        comb_inf = union_sorted(c_inf, new_inf)
                        w = 1. - (len(groups[-1])+len(groups[c])) * (len(comb_inf)/n_samples) \
                            + len(groups[-1]) * (len(new_inf)/n_samples) + len(groups[c]) * (len(c_inf)/n_samples)
                        
                        if w > 0:
                            heapq.heappush(heap, (-w, (c, idx_new)))
                                
    non_empty_groups = []   
    for g in groups:
        if len(g) > 0:
            non_empty_groups.append(g)
            
    return non_empty_groups

def _kernighan_lin_sweep(edges, side):
    """
    This is a modified form of Kernighan-Lin, which moves single nodes at a
    time, alternating between sides to keep the bisection balanced.  We keep
    two min-heaps of swap costs to make optimal-next-move selection fast.
    """
    costs0, costs1 = costs = BinaryHeap(), BinaryHeap()
    for u, side_u, edges_u in zip(count(), side, edges):
        cost_u = sum(w if side[v] else -w for v, w in edges_u)
        costs[side_u].insert(u, cost_u if side_u else -cost_u)

    def _update_costs(costs_x, x):
        for y, w in edges[x]:
            costs_y = costs[side[y]]
            cost_y = costs_y.get(y)
            if cost_y is not None:
                cost_y += 2 * (-w if costs_x is costs_y else w)
                costs_y.insert(y, cost_y, True)

    i = totcost = 0
    while costs0 and costs1:
        u, cost_u = costs0.pop()
        _update_costs(costs0, u)
        v, cost_v = costs1.pop()
        _update_costs(costs1, v)
        totcost += cost_u + cost_v
        yield totcost, i, (u, v)

#from networkx
def kernighan_lin_bisection(G, partition=None, max_iter=10, weight="weight", seed=None):
    """Partition a graph into two blocks using the Kernighan–Lin
    algorithm.

    This algorithm partitions a network into two sets by iteratively
    swapping pairs of nodes to reduce the edge cut between the two sets.  The
    pairs are chosen according to a modified form of Kernighan-Lin, which
    moves node individually, alternating between sides to keep the bisection
    balanced.

    Parameters
    ----------
    G : graph

    partition : tuple
        Pair of iterables containing an initial partition. If not
        specified, a random balanced partition is used.

    max_iter : int
        Maximum number of times to attempt swaps to find an
        improvemement before giving up.

    weight : key
        Edge data key to use as weight. If None, the weights are all
        set to one.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Only used if partition is None

    Returns
    -------
    partition : tuple
        A pair of sets of nodes representing the bipartition.

    Raises
    -------
    NetworkXError
        If partition is not a valid partition of the nodes of the graph.

    References
    ----------
    .. [1] Kernighan, B. W.; Lin, Shen (1970).
       "An efficient heuristic procedure for partitioning graphs."
       *Bell Systems Technical Journal* 49: 291--307.
       Oxford University Press 2011.

    """
    n = len(G)
    labels = list(G)
    if seed is not None:
        seed.shuffle(labels)
    index = {v: i for i, v in enumerate(labels)}

    if partition is None:
        side = [0] * (n // 2) + [1] * ((n + 1) // 2)
    else:
        try:
            A, B = partition
        except (TypeError, ValueError) as e:
            raise nx.NetworkXError("partition must be two sets") from e
        if not is_partition(G, (A, B)):
            raise nx.NetworkXError("partition invalid")
        side = [0] * n
        
        for a in A:
            side[a] = 1
      
    if G.is_multigraph():
        edges = [
            [
                (index[u], sum(e.get(weight, 1) for e in d.values()))
                for u, d in G[v].items()
            ]
            for v in labels
        ]
    else:
        edges = [
            [(index[u], e.get(weight, 1)) for u, e in G[v].items()] for v in labels
        ]

    updated = False
    for i in range(max_iter):
        costs = list(_kernighan_lin_sweep(edges, side))
        
        if len(costs) > 0:
            min_cost, min_i, _ = min(costs)
        else:
            break
            
        if min_cost >= 0:
            break

        for _, _, (u, v) in costs[: min_i + 1]:
            
            if side[u] == 0 or side[v] == 1:
                updated = True
            
            side[u] = 1
            side[v] = 0
    
    A = {u for u, s in zip(labels, side) if s == 0}
    B = {u for u, s in zip(labels, side) if s == 1}
    return A, B, updated
                    
def topol_kernighan_lin(G, group_size, n_iter=0, weight='weight', initial_groups=None):
    '''
        Classical kernigan lin heuristic applied to the group testing problem.
        https://en.wikipedia.org/wiki/Kernighan–Lin_algorithm
        
        We run the networkx implementation of KL for pairs of partitions
        in a given number of iterations (n_iter).
        
        If initial groups are not given, apply greedy-group.
    '''
    groups = []
    group_index = {}
    
    if n_iter == 0:
        n_iter = G.number_of_edges()
    
    if initial_groups is None:
        initial_groups = topol_greedy_group(G, group_size, weight=weight)
    
    new_vertex = len(G)
    ext_G = G.copy()
    modified = []
    for g in range(len(initial_groups)):
        modified.append(-1)
        initial_groups[g] = list(initial_groups[g])
        
        while len(initial_groups[g]) < group_size:
            initial_groups[g].append(new_vertex)
            ext_G.add_node(new_vertex)
            new_vertex = new_vertex + 1
        
        groups.append(list(initial_groups[g]))
     
    for i in range(n_iter):
        updated_iter = False
        for gi in range(len(groups)):
            for gj in range(gi+1, len(groups)):
                
                if modified[gi] < i-1 and modified[gj] < i-1:
                    break
                
                subgraph = nx.Graph()
                idx = {}
                inv_idx = []
                idx_gi = []
                idx_gj = []
                
                i = 0
                for v in groups[gi]:
                    idx_gi.append(i)
                    idx[v] = i
                    inv_idx.append(v)
                    subgraph.add_node(i)
                    i = i + 1
                    
                for v in groups[gj]:
                    idx_gj.append(i)
                    idx[v] = i
                    inv_idx.append(v)
                    subgraph.add_node(i)
                    i = i + 1
                
                for e in G.edges():
                    if (e[0] in groups[gi] or e[0] in groups[gj]) and (e[1] in groups[gi] or e[1] in groups[gj]):
                        if weight is None:
                            subgraph.add_edge(idx[e[0]],idx[e[1]])
                        else:
                            subgraph.add_edge(idx[e[0]],idx[e[1]], weight=G.edges[e]['weight'])
                
                new_gi, new_gj, updated = kernighan_lin_bisection(subgraph, 
                    partition=[idx_gi, idx_gj], max_iter=n_iter, weight=weight)
                
                if updated is True: 
                    groups[gi] = []
                    groups[gj] = []
                    
                    for v in new_gi:
                        groups[gi].append(inv_idx[v])
                    
                    for v in new_gj:
                        groups[gj].append(inv_idx[v])
                    
                    updated_iter = True
                    modified[gi] = i
                    modified[gj] = i
                    
        if updated_iter is False:
            break

    non_empty_groups = []
    for g in range(len(groups)):
        for i in range(len(G),new_vertex+1):
            if i in groups[g]:
                groups[g].remove(i)
                
        if len(groups[g]) > 0:        
            non_empty_groups.append(groups[g])
            
    return cluster_groups(non_empty_groups, group_size)

def samp_swap(gi, gj, groups, group_infec_lists, vert_infec_lists, infec_matrix, max_size, n_iter):
    '''
    	Performs swap operations between groups gi and gj
    '''
    gi_inf = group_infec_lists[gi]
    gj_inf = group_infec_lists[gj]

    group_i = list(groups[gi])
    group_j = list(groups[gj])

    n_samples = infec_matrix.shape[1]
    updated = False

    for i in range(n_iter):
        best_score = 0
        best_pair = None

        n_tests = len(group_i) * len(gi_inf) + len(group_j) * len(gj_inf)

        for v in group_i:
            v_inf = vert_infec_lists[v]
            for u in group_j:
                u_inf = vert_infec_lists[u]

		#Infections for group i removing v and adding u
                gi_mv_pu = diff_sorted(union_sorted(gi_inf, u_inf), v_inf)

		#Infections for group j removing u and adding v
                gj_mu_pv = diff_sorted(union_sorted(gj_inf, v_inf), u_inf)

		#Number of tests if u and v are swapped
                new_n_tests = len(group_i) * len(gi_mv_pu) + len(group_j) * len(gj_mu_pv)

                score = n_tests - new_n_tests

                if score > best_score:
                    best_pair = (u,v)
                    best_score = score

            if len(group_j) < max_size:
                gi_mv = diff_sorted(gi_inf, v_inf)
                gj_pv = union_sorted(gj_inf, v_inf)

                new_n_tests = (len(group_i)-1) * len(gi_mv) + (len(group_j)+1) * len(gj_pv)

                score = n_tests - new_n_tests

                if score > best_score:
                    best_pair = (None,v)
                    best_score = score

        if len(group_i) < max_size:
            for u in group_j:
                u_inf = vert_infec_lists[u]
                gi_pu = union_sorted(gi_inf, u_inf)
                gj_mu = diff_sorted(gj_inf, u_inf)
                new_n_tests = (len(group_i)+1) * len(gi_pu) + (len(group_j)-1) * len(gj_mu)

                score = n_tests - new_n_tests

                if score > best_score:
                    best_pair = (u,None)
                    best_score = score


        if best_score > 0:
            updated = True
            u,v = best_pair

            if u is not None:
                group_j.remove(u)
                group_i.append(u)

                u_inf = vert_infec_lists[u]

                gj_inf = diff_sorted(gj_inf, u_inf)
                gi_inf = union_sorted(gi_inf, u_inf)

            if v is not None:
                group_j.append(v)
                group_i.remove(v)

                v_inf = vert_infec_lists[v]

                gi_inf = diff_sorted(gi_inf, v_inf)
                gj_inf = union_sorted(gj_inf, v_inf)

        else:
            break

    return group_i, group_j, gi_inf, gj_inf, updated

def samp_kernighan_lin(G, group_size, infec_matrix, n_iter=0, initial_groups=None):
    '''
        Kernigan lin heuristic based on samples. Node swaps 
	minimize the expected number of tests.
        https://en.wikipedia.org/wiki/Kernighan–Lin_algorithm
    '''
    groups = []
    group_index = {}
    n_samples = infec_matrix.shape[1]

    if n_iter == 0:
        n_iter =  G.number_of_edges()

    vert_infec_lists = infection_lists(infec_matrix)

    if initial_groups is None:
        groups = samp_greedy_group(G, group_size, infec_matrix)
    else:
        groups = initial_groups

    group_infec_lists = []
    modified = []

    for g in range(len(groups)):
        group_infec_lists.append([])
        modified.append(-1)
        for v in groups[g]:
            group_infec_lists[-1] = union_sorted(group_infec_lists[-1], vert_infec_lists[v])

    for i in range(n_iter):
        updated_iter = False
        for gi in range(len(groups)):
            for gj in range(gi+1, len(groups)):

                if modified[gi] < i-1 and modified[gj] < i-1:
                    break

                new_gi, new_gj, new_infec_list_gi, new_infec_list_gj, updated \
                    = samp_swap(gi, gj, groups, group_infec_lists, vert_infec_lists, infec_matrix, group_size, n_iter)

                if updated:
                    updated_iter = True
                    groups[gi] = list(new_gi)
                    groups[gj] = list(new_gj)

                    group_infec_lists[gi] = new_infec_list_gi
                    group_infec_lists[gj] = new_infec_list_gj

                    modified[gi] = i
                    modified[gj] = i

        if updated_iter is False:
            break

    non_empty_groups = []
    for g in range(len(groups)):
        if len(groups[g]) > 0:
            non_empty_groups.append(groups[g])

    return non_empty_groups
