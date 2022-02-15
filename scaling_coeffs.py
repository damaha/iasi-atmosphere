#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 17:45:41 2017

@author: dmal
"""
import os, sys, json
import numpy as np

from generators import iasi_generator
import keras.backend as K

from data_load import get_datafiles
from generators import iasi_generator

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def update(existingAggregate, newValues):
    (means, M2s, count) = existingAggregate
    count += newValues.shape[0] 
    delta = newValues - np.repeat(np.expand_dims(means, 0), newValues.shape[0], axis=0) 
    means += (delta / count).sum(axis=0)
    delta2 = newValues - np.repeat(np.expand_dims(means, 0), newValues.shape[0], axis=0) 
    M2s += (delta * delta2).sum(axis=0)

    return (means, M2s, count)

# retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (mean, M2, count) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)
        
path = ".../IASI/d ata_v3/"

train_files, valid_files = get_datafiles(path)
indexs = [None, np.arange(47, 137).astype(int)]
train_generator = iasi_generator(train_files, batch_size=32, selected_channels=indexs)

nb_batchs = len(train_generator)
i=0
means, M2s, count = (np.zeros(4699), 
                     np.zeros(4699), 
                     0)
for inputs, _ in train_generator:
    print(str(i)+" out of "+str(nb_batchs))
    
    nVals = np.reshape(inputs, (inputs.shape[0]*inputs.shape[1]*inputs.shape[2], inputs.shape[3]))
    
    means, M2s, count = update((means, M2s, count), nVals)
    
    i += 1
    if i>=nb_batchs:
        break
means, variance, sample_variance = finalize((means, M2s, count))

json.dump(open("scaling_coeffs.json","w"),{"mean":means,"variance":variance})