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
        class_size = class_size.reshape(-1, 1).copy();
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
        class_size = class_size.reshape(-1, 1).copy();
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



def theil_One_L(mat:np.ndarray, base:str = 'e') -> float | np.ndarray:
    allowed_bases = ['e', '2', '10', 2, 10];
    if base not in allowed_bases:
        raise ValueError("'base' must be either of the following: '{}'.".format(
            "', '".join(allowed_bases[:3])));
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    mat1 = mat.copy();
    mat1[mat<0] = 0;
    n = (mat>0).astype(int).sum(axis = 0, keepdims = True);
    mu = mat1.sum(axis = 0, keepdims = True)/n;
    isIncl = (mat1>0).astype(int);
    mat1[mat1==0] = 2333;
    
    if base == 'e':
        THEIL = (isIncl*np.log(mu/mat1) / n).sum(axis = 0, keepdims = False);
    elif base == '2' or base == 2:
        THEIL = (isIncl*np.log2(mu/mat1) / n).sum(axis = 0, keepdims = False);
    elif base =='10' or base == 10:
        THEIL = (isIncl*np.log10(mu/mat1) / n).sum(axis = 0, keepdims = False);
    else:
        THEIL = "Meowth used pay day! It is super effective!";
    
    return THEIL;



def theil_One_T(mat:np.ndarray, base:str = 'e') -> float | np.ndarray:
    allowed_bases = ['e', '2', '10', 2, 10];
    if base not in allowed_bases:
        raise ValueError("'base' must be either of the following: '{}'.".format(
            "', '".join(allowed_bases[:3])));
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    mat1 = mat.copy();
    mat1[mat<0] = 0;
    n = (mat>0).astype(int).sum(axis = 0, keepdims = True);
    mu = mat1.sum(axis = 0, keepdims = True)/n;
    x = mat1/mu;
    x1 = x.copy();
    x1[x==0] = 3210;
    if base == 'e':
        THEIL = (x * np.log(x1) / n).sum(axis = 0, keepdims = False);
    elif base == '2' or base == 2:
        THEIL = (x * np.log2(x1) / n).sum(axis = 0, keepdims = False);
    elif base =='10' or base == 10:
        THEIL = (x * np.log10(x1) / n).sum(axis = 0, keepdims = False);
    else:
        THEIL = "Meowth used pay day! It is super effective!";
    
    return THEIL;





def theil_One_S(mat:np.ndarray, base:str = 'e') -> float | np.ndarray:
    allowed_bases = ['e', '2', '10', 2, 10];
    if base not in allowed_bases:
        raise ValueError("'base' must be either of the following: '{}'.".format(
            "', '".join(allowed_bases[:3])));
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    return (theil_One_L(mat, base) + theil_One_T(mat, base))/2.0;




def theil_byClass_L(mat:np.ndarray, 
                    class_size:np.ndarray,
                    base:str = 'e') -> float | np.ndarray:
    allowed_bases = ['e', '2', '10', 2, 10];
    if base not in allowed_bases:
        raise ValueError("'base' must be either of the following: '{}'.".format(
            "', '".join(allowed_bases[:3])));
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));

    if class_size.ndim > 2:
        raise ValueError("'class_size' must be either 1-d or 2-d array, but currently its dimension is {}.".format(class_size.ndim));
    if class_size.ndim == 1 and mat.ndim == 2:
        if len(class_size) != mat.shape[0]:
            raise ValueError("If 'class_size' is an 1-d array, it must have the same number of elements as the number of rows of 'mat'");
        class_size = class_size.reshape(-1, 1).copy();
    elif mat.shape!=class_size.shape:
        raise ValueError("When 'class_size' and 'mat' have the same dimensions, they must have the same shape. But currently the shape of 'mat' is {a} and the shape of 'class_size' is {b}.".format(a=mat.shape, b=class_size.shape));
    
    mat1 = mat.copy();
    w = class_size.copy();
    mat1[mat<0] = 0;
    w[w<0]=0;
    isIncl = (mat1>0).astype(int) * (w>0).astype(int);
    
    w = w * isIncl;
    mat1 = mat1 * isIncl;
    
    w = w/w.sum(axis = 0, keepdims = True);
    mu = (mat1 * w).sum(axis = 0, keepdims = True)
    
    mat1[mat1==0] = 2333;
    
    if base == 'e':
        THEIL = (np.log(mu/mat1) * w).sum(axis = 0, keepdims = False);
    elif base == '2' or base == 2:
        THEIL = (np.log2(mu/mat1) * w).sum(axis = 0, keepdims = False);
    elif base =='10' or base == 10:
        THEIL = (np.log10(mu/mat1) * w).sum(axis = 0, keepdims = False);
    else:
        THEIL = "Meowth used pay day! It is super effective!";
    
    return THEIL;




def theil_byClass_T(mat:np.ndarray, 
                    class_size:np.ndarray,
                    base:str = 'e') -> float | np.ndarray:
    allowed_bases = ['e', '2', '10', 2, 10];
    if base not in allowed_bases:
        raise ValueError("'base' must be either of the following: '{}'.".format(
            "', '".join(allowed_bases[:3])));
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));

    if class_size.ndim > 2:
        raise ValueError("'class_size' must be either 1-d or 2-d array, but currently its dimension is {}.".format(class_size.ndim));
    if class_size.ndim == 1 and mat.ndim == 2:
        if len(class_size) != mat.shape[0]:
            raise ValueError("If 'class_size' is an 1-d array, it must have the same number of elements as the number of rows of 'mat'");
        class_size = class_size.reshape(-1, 1).copy();
    elif mat.shape!=class_size.shape:
        raise ValueError("When 'class_size' and 'mat' have the same dimensions, they must have the same shape. But currently the shape of 'mat' is {a} and the shape of 'class_size' is {b}.".format(a=mat.shape, b=class_size.shape));
    
    mat1 = mat.copy();
    w = class_size.copy();
    mat1[mat<0] = 0;
    w[w<0]=0;
    isIncl = (mat1>0).astype(int) * (w>0).astype(int);
    
    w = w * isIncl;
    mat1 = mat1 * isIncl;
    
    w = w/w.sum(axis = 0, keepdims = True);
    mu = (mat1 * w).sum(axis = 0, keepdims = True)
    
    x = mat1/mu;
    x1 = x.copy();
    x1[x==0] = 3210;
    
    if base == 'e':
        THEIL = (x * np.log(x1) * w).sum(axis = 0, keepdims = False);
    elif base == '2' or base == 2:
        THEIL = (x * np.log2(x1) * w).sum(axis = 0, keepdims = False);
    elif base =='10' or base == 10:
        THEIL = (x * np.log10(x1) *w).sum(axis = 0, keepdims = False);
    else:
        THEIL = "Meowth used pay day! It is super effective!";
    
    return THEIL;




def theil_byClass_S(mat:np.ndarray, 
                    class_size:np.ndarray,
                    base:str = 'e') -> float | np.ndarray:
    allowed_bases = ['e', '2', '10', 2, 10];
    if base not in allowed_bases:
        raise ValueError("'base' must be either of the following: '{}'.".format(
            "', '".join(allowed_bases[:3])));
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));

    if class_size.ndim > 2:
        raise ValueError("'class_size' must be either 1-d or 2-d array, but currently its dimension is {}.".format(class_size.ndim));
    if class_size.ndim == 1 and mat.ndim == 2:
        if len(class_size) != mat.shape[0]:
            raise ValueError("If 'class_size' is an 1-d array, it must have the same number of elements as the number of rows of 'mat'");
    elif mat.shape!=class_size.shape:
        raise ValueError("When 'class_size' and 'mat' have the same dimensions, they must have the same shape. But currently the shape of 'mat' is {a} and the shape of 'class_size' is {b}.".format(a=mat.shape, b=class_size.shape));
    
    return (theil_byClass_T(mat, class_size, base) + theil_byClass_L(mat, class_size, base))/2.0;



def theil(mat:np.ndarray, 
          class_size:np.ndarray|None = None,
          method:str = 'L', 
          base:str = 'e') -> float | np.ndarray:
    
    '''
    Generate the Gini index from "unweighted" data, i.e. each element is 
    about just one entry. 
    
    Accept 1d or 2d array. When 2d array is used, values in each column
    belongs to a region.
    
    'method' must be one of the two:
       *  "L" (headcount weighting, default), or 
       -  "T" (income share weighting). 
       -  "S" (middleman).
    '''
    allowed_methods = ['T','L','S'];
    if method not in allowed_methods:
        raise ValueError("'method' must be either of the following: '{}'.".format(
            "', '".join(allowed_methods)));
    allowed_bases = ['e', '2', '10', 2, 10];
    if base not in allowed_bases:
        raise ValueError("'base' must be either of the following: '{}'.".format(
            "', '".join(allowed_bases[:3])));
    if mat.ndim > 2:
        raise ValueError("'mat' must be either 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    
    if class_size is None:
        if method == 'L':
            THEIL = theil_One_L(mat);
        elif method == 'T':
            THEIL = theil_One_T(mat);
        elif method =='S':
            THEIL = theil_One_S(mat);
        else:
            THEIL = "Doge coin is not healthy, it makes dog hungry!";
    else:
        if class_size.ndim > 2:
            raise ValueError("'class_size' must be either 1-d or 2-d array, but currently its dimension is {}.".format(class_size.ndim));
        if class_size.ndim == 1 and mat.ndim == 2:
            if len(class_size) != mat.shape[0]:
                raise ValueError("If 'class_size' is an 1-d array, it must have the same number of elements as the number of rows of 'mat'");
        elif mat.shape!=class_size.shape:
            raise ValueError("When 'class_size' and 'mat' have the same dimensions, they must have the same shape. But currently the shape of 'mat' is {a} and the shape of 'class_size' is {b}.".format(a=mat.shape, b=class_size.shape));
        
        if method == 'L':
            THEIL = theil_byClass_L(mat, class_size, base);
        elif method == 'T':
            THEIL = theil_byClass_T(mat, class_size, base);
        elif method == 'S':
            THEIL = theil_byClass_S(mat, class_size, base);
        else:
            THEIL == "Perrserker is guilty, it makes Meowth angry!";
    
    return THEIL;


