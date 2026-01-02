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

