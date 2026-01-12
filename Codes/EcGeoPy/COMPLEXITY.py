#!/usr/bin/python3.11
# -*- coding: utf-8 -*-


import numpy as np
from .RCA import rca

def rel_asymmetric(mat:np.ndarray, weight:np.ndarray|None = None) -> np.ndarray:
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
    
       weight: numpy 1-d array.  Optional. 
               Importance weight of each region (e.g. population, gdp size, etc). 
               Must be positive and dimension corresponds to the number of regions in mat.
    '''
    if weight is not None:
        if weight.ndim>1:
            raise ValueError("If 'weight' is supplied, it must be an 1-d array with the same number of elements like the number of columns of 'mat', but currently its dimension is {}.".format(weight.ndim));
        if len(weight) != mat.shape[1]:
            raise ValueError("'weight' must have the same number of elements like the number of columns of 'mat', but currently it has {} elements.".format(weight.ndim));
        if weight.min()<=0:
            raise ValueError("'weight' must be positive.");
    
    if weight is None:
        weight = 1;
    
    hasRCA = (mat>=1.0).astype(int);
    denominator = (hasRCA * weight).sum(axis = 1, keepdims = True);
    denominator[denominator == 0] = 10086;
    numerator = np.matmul(hasRCA * weight, hasRCA.T);
    Phi_asym = numerator / denominator;
    np.fill_diagonal(Phi_asym, 0);
    return Phi_asym;


def rel_symmetric(mat:np.ndarray, weight:np.ndarray|None = None) -> np.ndarray:
    '''Hidalgo's symetric version of relatedness between two items,
       phi(i,j) = min( phi(i->j), phi(j->i) )

       Parameters:
       -----
       mat: numpy 2-d array, either the value of RCA or whether the region 
            has RCA (i.e. 1/0) in each product/task. 
            Row: Product/Task/ etc.
            Col: Region
       
       weight: numpy 1-d array.  Optional. 
            Importance weight of each region (e.g. population, gdp size, etc). 
            Must be positive and dimension corresponds to the number of regions in mat.
    '''
    Phi_asym = rel_asymmetric(mat, weight);
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
                method: str = 'Symmetric',
                weight: np.ndarray|None = None) -> np.ndarray:
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
    
    weight: numpy 1-d array.  Optional. 
            Importance weight of each region (e.g. population, gdp size, etc). 
            Must be positive and dimension corresponds to the number of regions in mat.
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
    
    if weight is not None:
        if weight.ndim>1:
            raise ValueError("If 'weight' is supplied, it must be an 1-d array with the same number of elements like the number of columns of 'mat', but currently its dimension is {}.".format(weight.ndim));
        if len(weight) != mat.shape[1]:
            raise ValueError("'weight' must have the same number of elements like the number of columns of 'mat', but currently it has {} elements.".format(weight.ndim));
        if weight.min()<=0:
            raise ValueError("'weight' must be positive.");
        
    
    if input_type == 'Export':
        mat = rca(mat);
    
    if method == "Asymmetric":
        Result = rel_asymmetric(mat, weight);
    elif method == "Symmetric":
        Result = rel_symmetric(mat, weight);
    else:
        hasRCA = (mat>=1.0).astype(int);
        if weight is None:
            weight = 1;
        co_occur_counts = np.matmul(hasRCA*weight, hasRCA.T);
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



def rescale(x:np.ndarray) ->np.ndarray:
    
    eth = x[np.isfinite(x)].min()
    btc = x[np.isfinite(x)].max()
    if eth == btc:
        return x*0;
    else:
        return 100*(x-eth)/(btc-eth);



def pci_reflex(mat_RCA:np.ndarray, steps:int) -> np.ndarray:
    '''
    Compute complexity index of nodes using the reflection method.
    
    Parameters
    -----
    mat_RCA: numpy 2-d array.
        Row: Product/Task/ etc.
        Col: Region
    
    steps: integer. How many steps of reflection to be taken. 
        Max = 25.    
    '''
    if mat_RCA.ndim != 2:
        raise ValueError("'mat_RCA' must be a 2-d array, but currently its dimension is {}.".format(mat_RCA.ndim));
    if steps < 0:
        raise ValueError("'steps' must be an non-negative integer.");
    if steps > 25:
        print("[WARNING] Maximum 'steps' capped at 25.\n");
        steps = 25;
    
    hasRCA = (mat_RCA>=1.0).astype(int);
    if hasRCA.sum()==0:
        raise ValueError("Ensure that there must be some industry/region having RCA larger than 1 in the 'mat_RCA'.");
    
    diversity = hasRCA.sum(0, keepdims = True);
    ubiquity = hasRCA.sum(1, keepdims = True);
    
    if steps == 0:
        return rescale(ubiquity[:,0]);
    
    d0 = diversity;
    u0 = ubiquity;
    d1 = d0;
    u1 = u0;
    while steps > 0:
        steps -= 1;
        dd = (hasRCA*u1).sum(0, keepdims=True) / d0;
        dd[np.isnan(dd)]=0;
        uu = (hasRCA*d1).sum(1, keepdims=True) / u0;
        uu[np.isnan(uu)]=0;
        d1 = dd;
        u1 = uu;
    
    return rescale(u1[:,0]);


def eci_reflex(mat_RCA:np.ndarray, steps:int) -> np.ndarray:
    '''
    Compute economic complexity index of regions using the reflection method.
    
    Parameters
    -----
    mat_RCA: numpy 2-d array.
        Row: Product/Task/ etc.
        Col: Region
    
    steps: integer. How many steps of reflection to be taken. 
        Max = 25.    
    '''
    if mat_RCA.ndim != 2:
        raise ValueError("'mat_RCA' must be a 2-d array, but currently its dimension is {}.".format(mat_RCA.ndim));
    if steps < 0:
        raise ValueError("'steps' must be an non-negative integer.");
    if steps > 25:
        print("[WARNING] Maximum 'steps' capped at 25.\n");
        steps = 25;
    
    return pci_reflex(mat_RCA.T, steps);



def pci_eig(mat_RCA: np.ndarray) -> np.ndarray:
    '''
    Compute complexity index of nodes using the eigenfactor method.
    Input: numpy 2-d array.
        Row: Product/Task/ etc.
        Col: Region
    
    '''
    hasRCA = (mat_RCA>=1.0).astype(int);
    if hasRCA.sum()==0:
        raise ValueError("Ensure that there must be some industry/region having RCA larger than 1 in the 'mat_RCA'.");
    
    diversity = hasRCA.sum(0, keepdims = True);
    ubiquity = hasRCA.sum(1, keepdims = True);
    
    # problem_div = (diversity<=0);
    problem_ubi = (ubiquity<=0);
    diversity[diversity<=0] = 9988;
    ubiquity[ubiquity<=0] = 9992;
    
    doge = hasRCA/ubiquity;
    coin = hasRCA/diversity;
    dogecoin = np.matmul(doge, coin.T);
    
    eigenvalues, eigenvectors = np.linalg.eig(dogecoin);
    
    # Ensure sequence and get the second largest
    idx = np.argsort(eigenvalues)[::-1];
    PCI = eigenvectors[:,idx[1]];
    PCI[problem_ubi[:,0]>0] = np.nan;
    PCI = rescale(PCI.real);
    
    # Flipping signs to be consistent to the reflection outcome
    # PCIe = pci_reflex(mat_RCA, 25);
    # if np.corrcoef(PCI, PCIe)[0,1] < 0.0:
    #    PCI = -PCI;
    
    return PCI;



def eci_eig(mat_RCA: np.ndarray) -> np.ndarray:
    '''
    Compute complexity index of regions using the eigenfactor method.
    Input: numpy 2-d array.
        Row: Product/Task/ etc.
        Col: Region
    
    '''
    return pci_eig(mat_RCA.T);




def pci(mat: np.ndarray,
        input_type: str = 'Export',
        method: str = 'Eigenvector',
        steps: int | None = None ) -> np.ndarray:
    '''
    Compute the "product" complexity index of each node (e.g. product, task etc). 
    
    Parameters
    -----
    mat: numpy 2-d array. Either export of each region, or RCAs.
        Row: Product/Task/ etc.
        Col: Region
    
    input_type: String variable indicating input type, must be one of the two
                * "Export" => regional export data (Default)
                - "RCA" => Value of RCAs
                All other values will trigger an error.
    
    method: String variable indicating method of computation of complexity,
            must be either one of the two:
            * "Eigenvector" (default)
            - "Reflection"
            All other values will trigger an error. When the "Reflection" 
            method is used, one must also supply the number of steps in
            the reflection. 
    steps: Integer. Only useful when 'method' is "Eigenvector". 
           Maximum allowed steps set to 25. Otherwise you'll get a warning and
           the algorithm stops at step 25...
    '''
    if mat.ndim != 2:
        raise ValueError("'mat' must be a 2-d array, but currently its dimension is {}.".format(mat.ndim));
    allowed_methods = ['Eigenvector', 'Reflection'];
    allowed_types = ['Export', 'RCA'];
    
    if input_type not in allowed_types:
        raise ValueError("'input_type' must be one of: {}.".format("', '".join(allowed_types)));
    if method not in allowed_methods:
        raise ValueError("'method' must be one of: {}.".format("', '".join(allowed_methods)));
    if method == 'Reflection' and steps is None:
        raise ValueError("'steps' must be supplied when using the 'Reflection' method.");
    
    if input_type == 'Export':
        mat = rca(mat);
    
    
    if method == 'Eigenvector':
        if steps is not None:
            print("[WARNING] 'steps' ignored when using the 'Eigenvector' method.\n");
        PCI = pci_eig(mat);
    elif method == 'Reflection':
        PCI = pci_reflex(mat, steps);
    else:
        PCI = "When there is a will, perhaps there is still no way."
    
    return PCI;


        

def eci(mat: np.ndarray,
        input_type: str = 'Export',
        method: str = 'Eigenvector',
        steps: int | None = None ) -> np.ndarray:
    '''
    Compute the economic complexity index of each region.
    
    Parameters
    -----
    mat: numpy 2-d array. Either export of each region, or RCAs.
        Row: Product/Task/ etc.
        Col: Region
    
    input_type: String variable indicating input type, must be one of the two
                * "Export" => regional export data (Default)
                - "RCA" => Value of RCAs
                All other values will trigger an error.
    
    method: String variable indicating method of computation of complexity,
            must be either one of the two:
            * "Eigenvector" (default)
            - "Reflection"
            All other values will trigger an error. When the "Reflection" 
            method is used, one must also supply the number of steps in
            the reflection. 
    steps: Integer. Only useful when 'method' is "Eigenvector". 
           Maximum allowed steps set to 25. Otherwise you'll get a warning and
           the algorithm stops at step 25...
    '''
    if mat.ndim != 2:
        raise ValueError("'mat' must be a 2-d array, but currently its dimension is {}.".format(mat.ndim));
    allowed_methods = ['Eigenvector', 'Reflection'];
    allowed_types = ['Export', 'RCA'];
    
    if input_type not in allowed_types:
        raise ValueError("'input_type' must be one of: {}.".format("', '".join(allowed_types)));
    if method not in allowed_methods:
        raise ValueError("'method' must be one of: {}.".format("', '".join(allowed_methods)));
    if method == 'Reflection' and steps is None:
        raise ValueError("'steps' must be supplied when using the 'Reflection' method.");
    
    if input_type == 'Export':
        mat = rca(mat);
    
    
    if method == 'Eigenvector':
        if steps is not None:
            print("[WARNING] 'steps' ignored when using the 'Eigenvector' method.\n");
        ECI = eci_eig(mat);
    elif method == 'Reflection':
        ECI = eci_reflex(mat, steps);
    else:
        ECI = "When there is a will, perhaps there is still no way."
    
    return ECI;


def ci_calibrate(mat: np.ndarray, 
                   ref: np.ndarray) -> np.ndarray: 
    '''
    Calibrate the order in PCI and ECI indices.
    
    Parameters
    -----
    mat: numpy 1d or 2d array, containing PCI and ECI index
    
    ref: numpy 1d or 2d array. 
         "reference" of product and economic complexity with a known, desired order.
         It can be, e.g. PRODY index as reference for PCI, and EXPY index for ECI.
         If 'mat' is an 1-d array, 'ref' must be the same in shape.
         If 'mat' is an 2-d array, 'ref' must be either an 1-d array with the 
         same number of elements as the number of rows in 'mat', or having a same
         shape as 'mat'. 
    '''
    flag = True;
    if mat.ndim > 2:
        raise ValueError("'mat' must be a numpy 1-d or 2-d array, but currently its dimension is {}.".format(mat.ndim));
    if ref.ndim > 2:
        raise ValueError("'ref' must be a numpy 1-d or 2-d array, but currently its dimension is {}.".format(ref.ndim));
    if mat.ndim == 1:
        if mat.shape != ref.shape:
            raise ValueError("If 'mat' is a 1-d array, 'ref' must have exactly the same shape as 'mat'");
        m = mat.reshape(-1, 1);
    
    if mat.ndim == 2:
        if ref.ndim == 1:
            if len(ref) != mat.shape[0]:
                raise ValueError("If 'mat' is a 2-d array and 'ref' a 1-d array, the number of elements of 'ref' must be the same as the number of rows of 'mat'. But currently, 'mat' has {a} rows while 'ref' has {b} elements.".format(a = mat.shape[0], b = len(ref)));
        else:
            if ref.shape != mat.shape:
                raise ValueError("If 'mat' and 'ref' are 2-d arrays, they must have a same shape. But now the shape of 'mat' is {a} and 'ref' is {b}.".format(a = mat.shape, b = ref.shape));
            flag = False;
        m = mat;
    
    data = [];
    for i in range(m.shape[1]):
        mm = m[:, i];
        rr = ref if flag else ref[:, i];
        useful = np.isfinite(rr) & np.isfinite(mm);
        if np.sum(useful.astype(int))<2:
            data.append(mm.copy());
        else:
            sign = np.corrcoef(mm[useful], rr[useful])[0,1] >= 0;
            data.append(mm.copy() if sign else 100-mm.copy());
    
    if mat.ndim == 1:
        data = data[0];
    else:
        data = np.asarray(data).T;
    
    return data;



