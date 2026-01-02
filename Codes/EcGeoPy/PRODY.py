#!/usr/bin/python3.11
# -*- coding: utf-8 -*-


import numpy as np
from .RCA import rca;

def prody(mat: np.ndarray,
          val: np.ndarray,
          input_type: str = 'Export',
          weight: np.ndarray | None = None) -> np.ndarray:
    '''
    Generate Prody (alike) index
    
    Parameters
    -----
    mat: numpy 2-d array, either of the two:
          * export data
          - RCA values (default)
        Row: Product/Task/ etc.
        Col: Region
    
    val: numpy 1-d array.
         Measuring how great is each region (e.g. things like realgdppc)
         Dimension must correspond to the number of regions in mat.
    
    input_type: String variable indicating input type, must be one of the two
                * "Export" => regional export data (Default)
                - "RCA" => Value of RCAs
                All other values will trigger an error.
       
    weight: numpy 1-d array. Optional. 
            Importance weight of each region (e.g. population, gdp size, etc). 
            Dimension must correspond to the number of regions in mat.
    '''
    
    if mat.ndim != 2:
        raise ValueError("'mat' must be a 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    if val.ndim != 1:
        raise ValueError("'val' must be a 1-d array, but currently its dimension is {}.".format(val.ndim));
    
    if val.shape[0] != mat.shape[1]:
        raise ValueError("'val' must have the same number of elements as the number of regions implied by 'mat'");
    
    
    # Checking type of input
    allowed_types = ['RCA', 'Export'];
    if input_type not in allowed_types:
        raise ValueError("'input_type' must be one of: {}.".format(", ".join(allowed_types)));
    
    # Checking if weight is supplied, if is, then dimension must correspond to mat
    if weight is not None:
        if weight.ndim != 1:
            raise ValueError("'weight' must be a 1-d array, but currently its dimension is {}.".format(weight.ndim));
        if weight.shape[0] != mat.shape[1]:
            raise ValueError("'weight' must have the same number of elements as the number of regions implied by 'mat'");
    
    
    if input_type == 'Export':
        mat = rca(mat);
    
    if weight is not None:
        PRD = np.sum(mat * weight * val, 1) / np.sum(mat * weight, 1);
    else: 
        PRD = np.sum(mat * val, 1) / np.sum(mat, 1);
    
    return PRD;



def expy(exp_mat: np.ndarray,
         val: np.ndarray,
         weight: np.ndarray | None = None) -> np.ndarray:
    '''
    Generate EXPY (alike) index
    
    Parameters
    -----
    exp_mat: numpy 2-d array, exports of each region
             Row: Product/Task/ etc.
             Col: Region
    
    val: numpy 1-d array.
         Measuring how great is each region (e.g. things like realgdppc)
         Dimension must correspond to the number of regions in mat.
       
    weight: numpy 1-d array. Optional. 
            Importance weight of each region (e.g. population, gdp size, etc). 
            Dimension must correspond to the number of regions in mat.
    '''
    if exp_mat.ndim != 2:
        raise ValueError("'exp_mat' must be a 2-d array, but currently its dimension is {}.".format(exp_mat.ndim));
    
    if val.ndim != 1:
        raise ValueError("'val' must be a 1-d array, but currently its dimension is {}.".format(val.ndim));
    
    if val.shape[0] != exp_mat.shape[1]:
        raise ValueError("'val' must have the same number of elements as the number of regions implied by 'exp_mat'");
    
    # Checking if weight is supplied, if is, then dimension must correspond to mat
    if weight is not None:
        if weight.ndim != 1:
            raise ValueError("'weight' must be a 1-d array, but currently its dimension is {}.".format(weight.ndim));
        if weight.shape[0] != exp_mat.shape[1]:
            raise ValueError("'weight' must have the same number of elements as the number of regions implied by 'exp_mat'");
    
    if weight is not None:
        PRD = prody(exp_mat, val, weight = weight);
    else:
        PRD = prody(exp_mat, val);
    
    reg_basket = exp_mat / np.sum(exp_mat, 0, keepdims = True);

    EXPY = np.sum( reg_basket * PRD.reshape(-1,1) , 0);
    return EXPY;
