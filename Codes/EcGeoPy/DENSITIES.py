#!/usr/bin/python3.11
# -*- coding: utf-8 -*-

import numpy as np


def rel_density(relmat:np.ndarray, hasRCA:np.ndarray)->np.ndarray:
    '''
    Compute relatedness density.
    
    Parameters:
    -----
    relmat: numpy 2-d array. 
            Relatedness between each item. Must be symmetric.
    
    hasRCA: either numpy 1-d or 2-d array.
            Indicating whether one or multiple regions have already advantage
            in each of the items.
            Must be either boolean, or integers of 0 and 1.
            The shape (size of 1-d array, or the number of rows for 2-d array)
            must correspond to the shape of 'relmat'.
    '''
    if hasRCA.ndim > 2:
        raise ValueError("'hasRCA' must be either 1-d or 2-d array, but currently its dimension is {}.".format(hasRCA.ndim));
    
    hasRCA = hasRCA.astype(int);
    musashi = hasRCA.min();
    gojiroh = hasRCA.max();
    if musashi < 0 or gojiroh > 1:
        raise ValueError("Elements in 'hasRCA' must be either boolean, or integers of 0 and 1 only.");
    
    if relmat.ndim != 2:
        raise ValueError("'relmat' must a square 2-d array, but currently its dimension is {}.".format(relmat.ndim));
    
    if relmat.shape[0]!=relmat.shape[1]:
        raise ValueError("'relmat' must a square 2-d array, but currently its shape is {}.".format(relmat.shape));
    
    if hasRCA.ndim == 1:
        hasRCA = hasRCA.reshape(-1,1);

    if relmat.shape[0]!=hasRCA.shape[0]:
        raise ValueError("The number of elements or number of rows of 'hasRCA' must be the same as the number of rows of 'relmat'.");
    
    useful_relatedness = np.matmul(hasRCA.T, relmat);
    total_relatedness  = relmat.sum(axis = 0, keepdims = 1);
    
    RD = useful_relatedness/total_relatedness;
    if RD.ndim==2:
        RD = RD.T;
    
    return RD;
    

def compl_rel_density(relmat:np.ndarray, hasRCA:np.ndarray, redundant_items:np.ndarray) ->np.ndarray:
    '''
    Compute relatedness density.
    
    Parameters:
    -----
    relmat: numpy 2-d array. 
            Relatedness between each item. Must be symmetric.
    
    hasRCA: either numpy 1-d or 2-d array.
            Indicating whether one or multiple regions have already advantage
            in each of the items.
            Must be either boolean, or integers of 0 and 1.
            The shape (size of 1-d array, or the number of rows for 2-d array)
            must correspond to the shape of 'relmat'.
    
    redundant_items: either numpy 1-d or 2-d array.
            Indicating the redundant items to be discarded when computing
            relatedness density. 
            If an 1-d array is supplied, it must have the same shape as
            'hasRCA' in case it is also an 1-d array, or have the same number
            of element as the number of rows in 'hasRCA'. 
            If a 2-d array is supplied, its shape must be the same as 'hasRCA'.
    '''
    if hasRCA.ndim > 2:
        raise ValueError("'hasRCA' must be either 1-d or 2-d array, but currently its dimension is {}.".format(hasRCA.ndim));
    
    hasRCA = hasRCA.astype(int);
    musashi = hasRCA.min();
    gojiroh = hasRCA.max();
    if musashi < 0 or gojiroh > 1:
        raise ValueError("Elements in 'hasRCA' must be either boolean, or integers of 0 and 1 only.");
    
    if relmat.ndim != 2:
        raise ValueError("'relmat' must a square 2-d array, but currently its dimension is {}.".format(relmat.ndim));
    
    if relmat.shape[0]!=relmat.shape[1]:
        raise ValueError("'relmat' must a square 2-d array, but currently its shape is {}.".format(relmat.shape));
    
    if hasRCA.ndim == 1:
        hasRCA = hasRCA.reshape(-1, 1);
        if redundant_items.dims>1:
            raise ValueError("When 'hasRCA' is 1-d array, 'redundant_items' be an 1-d array with the same number of elements.");
        if len(hasRCA)!=len(redundant_items):
            raise ValueError("When 'hasRCA' is 1-d array, 'redundant_items' be an 1-d array with the same number of elements.");
        redundant_items=redundant_items.reshape(-1, 1);
        
    if relmat.shape[0]!=hasRCA.shape[1]:
        raise ValueError("The number of elements or number of rows of 'hasRCA' must be the same as the number of rows of 'relmat'.");
    
    if redundant_items.shape[1]>1 and redundant_items.shape!=hasRCA.shape:
        raise ValueError("When 'redundant_items' is a 2-d array, its shape must be the same as 'hasRCA'.");
    
    validAdv = hasRCA*(1 - redundant_items);
    return rel_density(relmat, validAdv);
