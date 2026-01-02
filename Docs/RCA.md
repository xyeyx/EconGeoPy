# RCA.py
<hr>

## rca

Compute revealed comparative advantage (i.e. the Balassa Index). 
<br/>


**Inputs**

* *exp_mat*: a 2-d numpy array, each row denotes the exported product and each column the regions.

**Return**

a 2-d numpy array, containing the RCA indices.

<br/>
<br/>

## isRCA

Indicating whether the region has RCA in each of the product.
<br/>

**Inputs**

* *exp_mat*: a 2-d numpy array, each row denotes the exported product and each column the regions.

* *isBoolean*: indicating if the output should be Boolean (set it to True), or 0/1 (set it to False). This is an optional parameter and its default value is False.


** Return **

a 2-d numpy array, containing the True/False flags or 0/1 integers indicating if each region possesses the comparative advantages.


