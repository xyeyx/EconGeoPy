#!/usr/bin/python3.11
# -*- coding: utf-8 -*-

import numpy as np
import EcGeoPy

Exports = np.asarray([[100, 300, 500,  800],
                      [100, 200, 100, 3000],
                      [100, 100, 100,  300]]);


RealGDPpC = np.asarray([100, 80, 50, 30]);
CountrySize = np.asarray([1, 3, 5, 7]);

RCA = EcGeoPy.rca(Exports);

PRODY = EcGeoPy.prody(Exports, RealGDPpC);
PRODY_ALT = EcGeoPy.prody(RCA, RealGDPpC, input_type='RCA');  # Same as above.
PRODY_Weighted = EcGeoPy.prody(Exports, RealGDPpC, weight = CountrySize);

EXPY = EcGeoPy.expy(Exports, RealGDPpC);
EXPY_WeightedPrody = EcGeoPy.expy(Exports, RealGDPpC, weight = CountrySize);

income = np.asarray([10, 20, 30, 40, 50]);
income_two = np.asarray([[10, 20, 30, 40, 50],
                         [20, 30, 40, 50, 10]]).T;

income_person  = np.asarray([10, 10, 10, 5, 1]) 
income_person2 = np.asarray([[10, 10, 10, 5, 1],
                             [10, 10,  5, 1,10]]).T;

EcGeoPy.robin_hood(income)
EcGeoPy.robin_hood(income_two)
EcGeoPy.robin_hood(income, income_person)
EcGeoPy.robin_hood(income_two, income_person2)

EcGeoPy.gini(income)
EcGeoPy.gini(income_two)
EcGeoPy.gini(income, income_person)
EcGeoPy.gini(income_two, income_person2)


sale_volumns = np.asarray([10, 50, 25, 75, 35, 5]);

# https://ts2.tech/en/top-10-web-browsers-of-2025-features-security-market-share-performance-comparison/
google_is_evil = np.asarray([66.5,    # Chrome & Friends Co. 
                             17.5,    # Safari,
                              5,      # Edge,
                              2.5,    # Firefox,
                              2,      # Opera,
                              2.5,    # Samsung,
                              1.5,    # Brave,
                              1,      # Vivaldi,
                              1,      # Sina,
                              0.3,    # Yandex
                              0.2     # Other
    ]);

EcGeoPy.herfindahl(sale_volumns);
EcGeoPy.herfindahl(google_is_evil);

