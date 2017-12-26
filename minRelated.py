#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:14:08 2017

@author: apple
"""

def isSeperated(a, b, adj):
    new_adj = np.copy(adj)
    new_adj[a:b,:] = 0
    new_adj[:,a:b] = 0
        
    visited = []
    stack = [b]
    while(len(stack) > 0):
        current_node = stack[-1]
        for i in range(new_adj.shape[1]):
            if (new_adj[current_node][i] == 1 and i not in visited):
                stack.append(i)
                if (i < a):
                    return False
        visited.append(current_node)
        stack.remove(current_node)
    return True

def main():
    
    adj = np.array(
            [[0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 1, 0, 0],
             [1, 0, 0, 1, 1, 0, 0, 0],
             [0, 1, 1, 0, 1, 0, 0, 0],
             [0, 0, 1, 1, 0, 1, 1, 1],
             [0, 1, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0]])
    
    min_related_nodes = np.zeros(adj.shape[0], dtype=np.float32)
    min_related_nodes[0] = 0
    for i in range(adj.shape[0]):
        for j in np.arange(i-1, -1, -1):
            if isSeperated(j, i, adj):
                min_related_nodes[i] = j
                break
            
    np.savez('asia_modelinfo.npz', min_related_nodes=min_related_nodes)
    
if __name__=='__main__':
    main()