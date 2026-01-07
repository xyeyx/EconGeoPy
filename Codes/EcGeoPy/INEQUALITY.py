#!/usr/bin/python3.11
# -*- coding: utf-8 -*-


import numpy as np



# Robin Hood
def rh_One(mat:np.ndarray) -> float | np.ndarray:
    '''
    Generate Robin Hood Index from "unweighted" data, i.e. each element is 
    about just one entry. 
    Accept 1d or 2d array. When 2d array is used, values in each column
    belongs to a country.
    '''
    
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    if mat.ndim == 1: 
        if int(mat[0]) == -19990930:
            print("How do you turn this on?");
            return "How do you turn this on?";
    
    x_mean = mat.mean(axis = 0);
    x_sum = mat.sum(axis =0);
    RH = 0.5 * np.sum(np.abs(mat - x_mean), axis = 0) / x_sum;
    
    return RH;


def rh_byClass(mat:np.ndarray, class_size:np.ndarray) -> float | np.ndarray:
    '''
    Generate the Robin Hood index based on info on different classes instead of
    individuals.
    mat => income by each class,   class_size => size of each class.
    The shape of the two must be the same.
    
    Accept 1d or 2d array. When 2d array is used, values in each column
    belongs to a country.
    '''
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    if class_size.ndim > 2:
        raise ValueError("'class_size' must be either 1-d or 2-d array, but currently its dimension is {}.".format(class_size.ndim));
    if class_size.ndim == 1 and mat.ndim == 2:
        if len(class_size) != mat.shape[0]:
            raise ValueError("If 'class_size' is an 1-d array, it must have the same number of elements as the number of rows of 'mat'");
        class_size = class_size.reshape(-1, 1);
    elif mat.shape!=class_size.shape:
        raise ValueError("When 'class_size' and 'mat' have the same dimensions, they must have the same shape. But currently the shape of 'mat' is {a} and the shape of 'class_size' is {b}.".format(a=mat.shape, b=class_size.shape));
    
    
    fairplay = class_size / class_size.sum(axis = 0);
    incomes = mat*class_size;
    realdist = incomes/incomes.sum(axis = 0);
    RH = 0.5*(np.abs(fairplay - realdist)).sum(0);
    
    return RH


def robin_hood(mat:np.ndarray, class_size:np.ndarray | None = None) -> float | np.ndarray:
    '''
    Generate the Robin Hood index.
    
    Parameters:
    -----
    mat: numpy array, either 1 or 2 dimensions.
         When input a two-dimensional array, each column is for data 
         in a same region.
    class_size: how large is the group with such income. This is an optional 
         input parameter. For instance, if 'mat' is [1,2,3] and 'class_size' is
         [3, 5, 2], it means that 3 persons have the income of 1, 5 persons 
         have 2, and 2 persons have 3. 
             If it is not supplied to the function, then it 
         treat each element in the 'mat' as equal in size (e.g. it can be then
         the income of each individual person). 
    '''
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    if class_size is None:
        return rh_One(mat);
    else:
        if class_size.ndim > 2:
            raise ValueError("'class_size' must be either 1-d or 2-d array, but currently its dimension is {}.".format(class_size.ndim));
        if class_size.ndim == 1 and mat.ndim == 2:
            if len(class_size) != mat.shape[0]:
                raise ValueError("If 'class_size' is an 1-d array, it must have the same number of elements as the number of rows of 'mat'");
        
        elif mat.shape!=class_size.shape:
            raise ValueError("When 'class_size' and 'mat' have the same dimensions, they must have the same shape. But currently the shape of 'mat' is {a} and the shape of 'class_size' is {b}.".format(a=mat.shape, b=class_size.shape));
        return rh_byClass(mat, class_size);



# Herfindahl(-Hirschman) index
def herfindahl(mat:np.ndarray) -> float | np.ndarray:
    '''
    Generate the Herfindahl-Hirschman index of market concentration.
    
    Parameters
    -----
    mat: numpy array, either 1 or 2 dimensions.
         When input a two-dimensional array, each column is for data 
         in a same region.
    '''
    mat2 = mat*mat;
    mat2sum = mat2.sum(axis = 0);
    matsum = mat.sum(axis = 0)
    matsum2 = matsum*matsum;
    HFD = mat2sum / matsum2;
    
    return HFD;



# Gini 
def gini_One(mat:np.ndarray) -> float | np.ndarray:
    '''
    Generate the Gini index from "unweighted" data, i.e. each element is 
    about just one entry. 
    Accept 1d or 2d array. When 2d array is used, values in each column
    belongs to a country.
    '''
    
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    seq = mat.argsort(axis = 0).argsort(axis = 0) + 1;
    n = len(mat);
    G = 2.0 * np.sum(mat*seq, axis = 0) / ( n * mat.sum(axis = 0) ) - (n+1)/n;
    
    return G;



# Gini 
def gini_byClass(mat:np.ndarray, class_size:np.ndarray) -> float | np.ndarray:
    '''
    Generate the Gini index based on info on different classes instead of
    individuals.
    mat => income by each class,   class_size => size of each class.
    The shape of the two must be the same.
    
    Accept 1d or 2d array. When 2d array is used, values in each column
    belongs to a country.
    '''
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    if class_size.ndim > 2:
        raise ValueError("'class_size' must be either 1-d or 2-d array, but currently its dimension is {}.".format(class_size.ndim));
    if class_size.ndim == 1 and mat.ndim == 2:
        if len(class_size) != mat.shape[0]:
            raise ValueError("If 'class_size' is an 1-d array, it must have the same number of elements as the number of rows of 'mat'");
        class_size = class_size.reshape(-1, 1);
    elif mat.shape!=class_size.shape:
        raise ValueError("When 'class_size' and 'mat' have the same dimensions, they must have the same shape. But currently the shape of 'mat' is {a} and the shape of 'class_size' is {b}.".format(a=mat.shape, b=class_size.shape));
    
    seq = mat.argsort(axis = 0);
    mat_asc = np.take_along_axis(mat, seq, axis = 0);
    cs_asc = np.take_along_axis(class_size, seq, axis = 0);
    csh_asc = cs_asc/cs_asc.sum(axis = 0);
    xfx = mat_asc*csh_asc;
    cumSum_xfx = xfx.cumsum(axis = 0);
    cumSum_xfx0 = np.insert(cumSum_xfx, 0, 0, axis = 0)[:-1];
    G = 1 - np.sum(csh_asc * (cumSum_xfx0 + cumSum_xfx),
                       axis = 0) / cumSum_xfx[-1];
    
    return G;




def gini(mat:np.ndarray, class_size:np.ndarray | None = None) -> float | np.ndarray:
    '''
    Generate the Gini index.
    
    Parameters:
    -----
    mat: numpy array, either 1 or 2 dimensions.
         When input a two-dimensional array, each column is for data 
         in a same region.
    class_size: how large is the group with such income. This is an optional 
         input parameter. For instance, if 'mat' is [1,2,3] and 'class_size' is
         [3, 5, 2], it means that 3 persons have the income of 1, 5 persons 
         have 2, and 2 persons have 3. 
             If it is not supplied to the function, then it 
         treat each element in the 'mat' as equal in size (e.g. it can be then
         the income of each individual person). 
    '''
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    if class_size is None:
        return gini_One(mat);
    else:
        if class_size.ndim > 2:
            raise ValueError("'class_size' must be either 1-d or 2-d array, but currently its dimension is {}.".format(class_size.ndim));
        if class_size.ndim == 1 and mat.ndim == 2:
            if len(class_size) != mat.shape[0]:
                raise ValueError("If 'class_size' is an 1-d array, it must have the same number of elements as the number of rows of 'mat'");
        
        elif mat.shape!=class_size.shape:
            raise ValueError("When 'class_size' and 'mat' have the same dimensions, they must have the same shape. But currently the shape of 'mat' is {a} and the shape of 'class_size' is {b}.".format(a=mat.shape, b=class_size.shape));
        return gini_byClass(mat, class_size);

