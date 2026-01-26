import numpy as np
import EcGeoPy as egp
import sqlite3

'''
The codes are just for illustrative purpose,... for illustrating the 
functionality of the EcGeoPy package in computing indices that are often
used in the field of economic geography.

While these are all from real-world data and therefore not faked ones,
note that the trade data is for 2024 and PWT national account data is 
for 2023, ... but should be close enough anyway ;-)

Since my laptop is too old (it is built back in 2011), and running the
popular python notebooks will often making my browser freezed up, the codes
are all "bare-bone" codes in texts. If you are using python notebooks, the
codes can be simply copy-pasted by each line or block into the browser 
environment. They also work well with other environments like spyder, idle,
M$FT studio, or just in console-alike places like kate / emacs.
'''


def GetSql(query, path):
    conn = sqlite3.connect(path);
    cursor = conn.cursor(); 
    cursor.execute(query);
    rows = cursor.fetchall();
    conn.close();
    return rows;


# Import databases
PWT_DB = 'data/PWT2023.db';
EXPORT_DB = 'data/HS2_Export2024.db';



# use the countries/regions with at least a million population, and are 
# common in both BACI and PWT datasets. 
cmd = '''SELECT countrycode FROM pwt2023 WHERE pop>=1''';

GDP_DATA = GetSql(cmd, PWT_DB);
iso3_PWT =  [x[0] for x in GDP_DATA];

cmd = '''SELECT iso3 from Iso3Info'''
iso3_BACI = [x[0] for x in GetSql(cmd, EXPORT_DB)];


iso3 = sorted(set(iso3_BACI) & set(iso3_PWT));

cmd = '''
SELECT countrycode, rgdpo/pop as rgdppc 
FROM pwt2023
WHERE countrycode in ("{iso3}")
ORDER BY countrycode
'''.format(iso3= '","'.join(iso3))
rgdppc = GetSql(cmd, PWT_DB);

cmd = '''
SELECT i.iso3, j.hs2, j.exp_value 
  FROM  Iso3Info i,
        Export2024 j
  WHERE i.origin_id = j.origin_id
    AND i.iso3 in ("{iso3}")
    ORDER BY i.iso3,j.hs2
'''.format(iso3= '","'.join(iso3))
exports = GetSql(cmd, EXPORT_DB);

# get the set of HS2 that have appeared in the data
hs2 = sorted(set(x[1] for x in exports));

# Get HS product names
cmd = '''SELECT hs2, name FROM HsInfo'''
hs2name = dict({x[0]:x[1].strip() for x in GetSql(cmd, EXPORT_DB)})


# Number of countries/regions and products
C_NUM = len(iso3);
P_NUM = len(hs2);


# Generate a mapping (i.e. sequence id) per each iso3 and HS codes.
cid = dict();
for i in range(C_NUM):
    cid.update({iso3[i]:i});

pid = dict();
for i in range(P_NUM):
    pid.update({hs2[i]:i});

# mapping from product sequence id to its name
pid2name = dict({pid[x]:hs2name[x] for x in pid});


# Generic preparation has finished.




# Matrix containing exports by each country/region's exports
ExpMat = np.zeros([P_NUM, C_NUM]);
for x in exports:
    ExpMat[ pid[x[1]] ,cid[x[0]] ] = x[2];


# RCA of each product in each country/region
RCA_Mat = egp.rca(ExpMat);

# List the countries with the highest and lowest RCA of ...
# Clock and Watches (HS code = 91, PID = 89)
hs_example = 91;
RCA_Watch = RCA_Mat[pid[hs_example], :];
RCA_Watch = [(iso3[cid], float(val)) for cid, val in enumerate(RCA_Watch)];
RCA_Watch.sort(key = lambda x: -x[1])
print('''
The country/regions with maximum and minimum RCAs in the HS2 Code {num}:
"{name}",
are the following ones:
------------------------------------
 #   TOP                BOTTOM
------------------------------------'''.format(num = hs_example, 
                                 name = hs2name[hs_example]));

for i in range(10):
    print("{num:2d}   {winner}  {vw:7.4f}       {loser}  {lw:7.4f}".format(
        num = i+1, 
        winner=RCA_Watch[i][0],
        loser =RCA_Watch[-1-i][0],
        vw = RCA_Watch[i][1],
        lw = RCA_Watch[-1-i][1]
        ));
print("-----------------------------------")

'''
Reasonable outcome: Switzerland is basically the world monopoly, while 
Mauritus, Hong Kong, Singapore etc are place where transactions of watches 
are rather active.  
'''





