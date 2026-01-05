#!/usr/bin/python3.11
# -*- coding: utf-8 -*-


import numpy as np
from .RCA import rca

def rel_asymmetric(mat:np.ndarray) -> np.ndarray:
    '''Hidalgo's asymetric version of relatedness between two items, based on
       the conditional probability of having both items i and j, conditioning
       on already having i. Namely...
                     Prob(RCA i and i)    
       phi(i->j) = ---------------------
                       Prob (RCA i)

       Parameters:
       -----
       mat: numpy 2-d array, either the value of RCA or whether the region 
            has RCA (i.e. 1/0) in each product/task. 
            Row: Product/Task/ etc.
            Col: Region
    '''
    hasRCA = (mat>=1.0).astype(int);
    denominator = hasRCA.sum(axis = 1, keepdims = True);
    denominator[denominator == 0] = 10086;
    numerator = np.matmul(hasRCA, hasRCA.T);
    Phi_asym = numerator / denominator;
    np.fill_diagonal(Phi_asym, 0);
    return Phi_asym;


def rel_symmetric(mat:np.ndarray) -> np.ndarray:
    '''Hidalgo's symetric version of relatedness between two items,
       phi(i,j) = min( phi(i->j), phi(j->i) )

       Parameters:
       -----
       mat: numpy 2-d array, either the value of RCA or whether the region 
            has RCA (i.e. 1/0) in each product/task. 
            Row: Product/Task/ etc.
            Col: Region
    '''
    Phi_asym = rel_asymmetric(mat);
    return np.minimum(Phi_asym, Phi_asym.T);


def jaccard_normalization(co_occur_mat:np.ndarray) -> np.ndarray:
    '''
    Jaccard normalisation
    Input: numpy 2-d array. Must have a square shape.
           Matrix for the count of co-occurance in i and j
    '''
    co0 = co_occur_mat.sum(axis = 0, keepdims = True);
    co1 = co_occur_mat.sum(axis = 1, keepdims = True);
    co0[co0==0] = 12345;
    co1[co1==0] = 54321;
    J = co_occur_mat/(co0 + co1 - co_occur_mat);
    np.fill_diagonal(J, 0);
    return J;

def cosine_normalization(co_occur_mat:np.ndarray) -> np.ndarray:
    '''
    Cosine similarity normalisation
    Input: numpy 2-d array. Must have a square shape.
           Matrix for the count of co-occurance in i and j
    '''
    co0 = co_occur_mat.sum(axis = 0, keepdims = True);
    co1 = co_occur_mat.sum(axis = 1, keepdims = True);
    co0[co0==0] = 12345;
    co1[co1==0] = 54321;
    Cosplay = co_occur_mat/np.sqrt(co0 * co1);
    np.fill_diagonal(Cosplay, 0);
    return Cosplay;

def ass_str_normalization(co_occur_mat:np.ndarray) -> np.ndarray:
    '''
    Association strength normalisation
    Input: numpy 2-d array. Must have a square shape.
           Matrix for the count of co-occurance in i and j
    '''
    co0 = co_occur_mat.sum(axis = 0, keepdims = True);
    co1 = co_occur_mat.sum(axis = 1, keepdims = True);
    T = co0.sum();
    co0[co0==0] = 12345;
    co1[co1==0] = 54321;
    Cosplay = T*co_occur_mat/(co0 * co1);
    np.fill_diagonal(Cosplay, 0);
    return Cosplay;    

def stijn_normalization(co_occur_mat:np.ndarray) -> np.ndarray:
    '''
    Steijn probability normalisation
    Input: numpy 2-d array. Must have a square shape.
           Matrix for the count of co-occurance in i and j
    '''
    co0 = co_occur_mat.sum(axis = 0, keepdims = True);
    co1 = co_occur_mat.sum(axis = 1, keepdims = True);
    T = co0.sum();
    ST = co_occur_mat/( ((co0/T)*(co1/(T-co0)) + 
                         (co1/T)*(co0/(T-co1)))*(T/2));
    ST[np.isnan(ST)]=0;
    ST[np.isinf(ST)]=0;
    return ST;

def relatedness(mat:np.ndarray, 
                input_type:str = 'Export',
                method: str = 'Symmetric') -> np.ndarray:
    '''
    Generate the relatedness matrix based on co-occurance of revealed
    comparative advantage. See eg. Hildalgo et al (2007) "The Product Space".
    
    
    Parameters:
    -----
    mat: numpy 2-d array, either of the two:
          * Export data (default)
          - RCA values
        Row: Product/Task/ etc.
        Col: Region
    
    input_type: String variable indicating input type, must be one of the two
                * "Export" => regional export data (Default)
                - "RCA" => Value of RCAs
                All other values will trigger an error.
    
    method: Method of computing. Currently supporting:
            * Symmetric => Symmetric version of conditional co-occurance probability (default)
            - Asymmetric => Asymmetric version of conditional co-occurance probability
            - Cosine => Cosine similarity 
            - Association => Association strength
            - Jaccard => Jaccard similarity
            - Steijn => Steijn's probability measure
    '''
    if mat.ndim != 2:
        raise ValueError("'mat' must be a 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    # Check type of method
    allowed_methods = ['Symmetric',
                      'Asymmetric',
                      'Cosine',
                      'Association',
                      'Jaccard',
                      'Steijn'];
    if method not in allowed_methods:
        raise ValueError("'method' must be one of: '{}'.".format("', '".join(allowed_methods)));
    
    # Checking type of input
    allowed_types = ['RCA', 'Export'];
    if input_type not in allowed_types:
        raise ValueError("'input_type' must be one of: {}.".format("', '".join(allowed_types)));
    
    if input_type == 'Export':
        mat = rca(mat);
    
    if method == "Asymmetric":
        Result = rel_asymmetric(mat);
    elif method == "Symmetric":
        Result = rel_symmetric(mat);
    else:
        hasRCA = (mat>=1.0).astype(int);
        co_occur_counts = np.matmul(hasRCA, hasRCA.T);
        np.fill_diagonal(co_occur_counts, 0);
        if method == "Jaccard":
            Result = jaccard_normalization(co_occur_counts);
        elif method == "Cosine":
            Result = cosine_normalization(co_occur_counts);
        elif method == "Association":
            Result = ass_str_normalization(co_occur_counts);
        elif method == "Steijn":
            Result = stijn_normalization(co_occur_counts);
        else:
            Result = "Cheese Steak Jimmy's";
    
    return Result;

