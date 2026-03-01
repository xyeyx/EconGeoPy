#!/usr/bin/python3.11
# -*- coding: utf-8 -*-

# Generate related and unrelated variety indicators
# Input => numpy 1-d or 2-d array of exports from each region
#              when 2-d array is supplied, the number of rows must
#              be the same as the elements 
#          numpy 1-d array of flags (classifying the clusters)
#          optional: numpy 1-d array of reference weights
#          dimension of the two (three) must correspond


import numpy as np;
from typing import Tuple, List, Any;
from .ENTROPY import entropy, kl;

def group_by_flag(mat: np.ndarray,
                  flag: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    if flag.ndim >1:
        raise ValueError("'flag' must be an 1-d array, but currently its dimension is {}.".format(flag.ndim));
    if mat.shape[0]!=flag.shape[0]:
        raise ValueError("The shapes of 'mat' and 'flag' must correspond, but currently their shapes are {a} and {b}.".format(mat.shape, flag.shape));
    
    unique_flag = np.unique(flag);
    if mat.ndim == 1:
        grouped = [mat[flag == doge] for doge in unique_flag];
    else:
        grouped = [mat[flag == doge,:] for doge in unique_flag];
    
    return (grouped, unique_flag);




def unrel_variety(mat: np.ndarray, 
                  flag: np.ndarray|List[Any], 
                  ref: None|np.ndarray = None) -> np.ndarray|float:
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    if flag.ndim >1:
        raise ValueError("'flag' must be an 1-d array, but currently its dimension is {}.".format(flag.ndim));
    if mat.shape[0]!=flag.shape[0]:
        raise ValueError("The shapes of 'mat' and 'flag' must correspond, but currently their shapes are {a} and {b}.".format(mat.shape, flag.shape));
    
    if ref is not None:
        if ref.ndim>1:
            raise ValueError("'ref' must be an 1-d array, but currently its dimension is {}.".format(ref.ndim));
        if ref.shape[0]!=mat.shape[0]:
            raise ValueError("The shapes of 'mat' and 'ref' must correspond, but currently their shapes are {a} and {b}.".format(mat.shape, ref.shape));
    
    (umat, uflag) = group_by_flag(mat, flag);
    mat_class = np.asarray([np.sum(x, axis = 0) for x in umat]);
    if ref is None:
        return entropy(mat_class);
    else:
        (uref, uflag) = group_by_flag(ref, flag);
        ref_class = np.asarray([np.sum(x) for x in uref]);
        return -kl(mat_class, ref_class);




def rel_variety(mat: np.ndarray, 
                  flag: np.ndarray|List[Any], 
                  ref: None|np.ndarray = None) -> np.ndarray|float:
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    if flag.ndim >1:
        raise ValueError("'flag' must be an 1-d array, but currently its dimension is {}.".format(flag.ndim));
    if mat.shape[0]!=flag.shape[0]:
        raise ValueError("The shapes of 'mat' and 'flag' must correspond, but currently their shapes are {a} and {b}.".format(mat.shape, flag.shape));
    
    if ref is not None:
        if ref.ndim>1:
            raise ValueError("'ref' must be an 1-d array, but currently its dimension is {}.".format(ref.ndim));
        if ref.shape[0]!=mat.shape[0]:
            raise ValueError("The shapes of 'mat' and 'ref' must correspond, but currently their shapes are {a} and {b}.".format(mat.shape, ref.shape));
    
    (umat, uflag) = group_by_flag(mat, flag);
    mat_class = np.asarray([np.sum(x, axis=0) for x in umat]);
    if mat.ndim ==1:
        wgt_class = mat_class/np.sum(mat_class, axis = 0, keepdims = True);
    else:
        wgt_class = mat_class/np.sum(mat_class, axis = 0);
    
    if ref is None:
        entropy_perClass = np.asarray([entropy(x) for x in umat]);
    else:
        (uref, uflag) = group_by_flag(ref, flag);
        entropy_perClass = np.asarray([0.0 - kl(x[0],x[1])   for x in zip(umat, uref)]);
    
    return np.sum(entropy_perClass * wgt_class, axis = 0);

