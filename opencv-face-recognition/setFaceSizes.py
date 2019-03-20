# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:09:10 2019

@author: em370_000
"""

import pickle
faceData = {'eric': 155, 'supriya': 155, 'ashwin': 155,'talia':155, 'nelson': 155,'unknown':155 }


with open('faceSizes.pickle', 'wb') as handle:
    pickle.dump(faceData, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('faceSizes.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(faceData == b)