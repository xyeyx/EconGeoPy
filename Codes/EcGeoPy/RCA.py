#!/usr/bin/python3.11
# -*- coding: utf-8 -*-

# Generate RCA/isRCA, from gross export data
# Input: numpy 2-d array of exports from each region, with:
#        Row: Product/Task/ etc.
#        Col: Region
#
#   For example
#         |  Ctry1   Ctry2   Ctry3   Ctry4
#   ------+----------------------------------
#   Prod1 |  
#   Prod2 |
#   Prod3 |
#   
#   Will be the shape of (3, 4)
#
# Output:Corresponding RCA matrix of the same shape.
#
# Negative exports will be corrected to zero. If there is a region with
# completely zero exports, or a product/task exported by no single region,
# the corresponding RCA value will be set to zero.


import numpy as np

def rca(exp_mat: np.ndarray) -> np.ndarray:
    '''
    Generate RCA from export data.
        
    parameters
    ----
    exp_mat : np.ndarray
              Must be 2-d dimension. 
              dim 0: product (i.e row)
              dim 1: region  (i.e. column) 
    '''
    
    if exp_mat.ndim != 2:
        raise ValueError("exp_mat must be a 2-d array, currently the input dimension is {}.".format(exp_mat.ndim));
    
    exp_mat[exp_mat<=0]=0;
    if np.issubdtype(exp_mat.dtype, np.integer):
        exp_mat = exp_mat.astype(float);
    
    reg_sum = np.sum(exp_mat, 0, keepdims = True);
    reg_sum[reg_sum<=0]=0.123;
    prod_sum = np.sum(exp_mat, 1, keepdims = True);
    grandtotal = np.sum(reg_sum);
    if grandtotal <= 0:
        raise ValueError("exp_mat has no positive values.");
    
    ExpWorldShare = prod_sum/grandtotal;
    ExpWorldShare[ExpWorldShare<=0]=0.123;
    
    RCA = (exp_mat/reg_sum) / ExpWorldShare;
    
    return RCA;




def isRCA(exp_mat: np.ndarray, 
          isBoolean: bool = False) -> np.ndarray:
    '''
    Return whether a region has comparative advantage in certain product.
        
    parameters
    ----
    exp_mat : np.ndarray
              Must be 2-d dimension. 
              dim 0: product (i.e. row)
              dim 1: region  (i.e. column) 
    
    isBoolean: bool
               Indicating whether the output should be T/F, or 0/1.
               Default value: False, output will be integer 0/1. Set to True 
    '''
    RCA = rca(exp_mat);
    hasRCA = (RCA >= 1.0);
    if isBoolean:
        return hasRCA
    else:
        return hasRCA.astype(int)


    
