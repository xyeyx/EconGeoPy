# PRODY.py
Prody & Expy indices as in Hausmann, Hwang and Rodrik (2007).
<hr>

## prody
Compute the PRODY index 
<br/>


**Inputs**

* *mat*: a 2-d numpy array, each row denotes the exported product and each column the regions. It can be either export data (the default setting), or RCA values.

* *val*: a 1-d numpy array, indicating how "great" is each region. The most often seen case is (real) GDP per capita in each country, as in the paper by Hausmann, Hwang and Rodrik. But it can be other things, and not necessarily to be "good" (e.g. it can be pollution intensity).

* *input_type*: a string. Can be either "Export" or "RCA", indicating the type of data used in the *mat* parameter.

* *weight*: a 1-d numpy array. This is an optional parameter. By default, the function compute the original PRODY index as in  If supplied, it will serve as an importance weight over each region, when calculating the PRODY index as in Hausmann, Hwang and Rodrik (2007). But sometimes it might be desired if if one attach more importance to a country larger in economic or population size. If so, one can place things like total GDP or population headcount of each country in this parameter.


**Return**

a 1-d numpy array, containing the PRODY indices of each product.

<br/>
<br/>

## expy

Compute the EXPY index
<br/>

**Inputs**

* *exp_mat*: a 2-d numpy array, each row denotes the exported product and each column the regions.

* *val*: a 1-d numpy array, indicating how "great" is each region. 

* *weight*: a 1-d numpy array. If supplied, the weighted PRODY index will be used when computing the EXPY index.

**Return**

a 1-d numpy array, containing the EXPY indices of each region.

