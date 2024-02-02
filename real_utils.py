import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
import numpy as np
from time import time
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from os import path, makedirs

import matplotlib.pyplot as plt

import h5py
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
import scipy.signal
from time import time
from tensorflow_addons.optimizers import LAMB
from itertools import combinations

import glob
import json
import multiprocessing

def normalize(strain):
    std = np.std(strain[:])
    strain[:] /= std
    return strain


# func. to make preds
def make_preds(whitened_L1, whitened_H1, whitened_V1, Model):

    # Load Strain
    normalized_L1 = normalize(whitened_L1)
    normalized_H1 = normalize(whitened_H1)
    normalized_V1 = normalize(whitened_V1)
    data = np.stack(( normalized_L1, normalized_H1,  normalized_V1), axis=1)

    # Create datagenerators
    dg_0 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=0, batch_size=256)
    delta = ((len(dg_0)-1)//2)*256
    dg_0 = dg_0[(len(dg_0)-1)//2][0]
    dg_5 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=2047, batch_size=256)
    dg_5 = dg_5[(len(dg_5)-1)//2][0]
    # Make preds
    preds_0 = Model.predict_generator(dg_0, verbose=1)
    preds_5 = Model.predict_generator(dg_5, verbose=1)
    
    return preds_0.ravel(), preds_5.ravel(), delta





# func. to find peaks
def find_peaks(preds, threshold = 0.9, width = [1500,3000], mean = 0.9):
    '''
    preds: 1D numpy array of sigmoid output from the NN
    '''
    test_p = preds
    
    peaks, properties =  scipy.signal.find_peaks(test_p, height=threshold, width = width, distance = 4096*1 )

    left = properties['left_ips']
    right = properties['right_ips']

    f_left = []
    f_right = []
    for i in range(len(left)):
        sliced = test_p[int(left[i]):int(right[i])] 
        if (np.mean(sliced>mean)>mean):
            f_left.append(int(left[i]))
            f_right.append(int(right[i]))
            
    return peaks, f_left, f_right


# func. to merge windows
def merge_windows(triggers_0, triggers_5, delta):

    triggers = {}
    
    for key in triggers_0.keys():

        right_0 = triggers_0[key]
        right_5 = triggers_5[key]

        combined = right_0.copy()
        
        for r_5 in right_5:
            keep = True
            for r_0 in right_0:
                if abs(r_5 - r_0) < 1/2:
                    keep = False
            if keep:
                combined.append(r_5)
        
        triggers[key] = [x + delta for x in combined]
    
    return triggers


# Convert right ips to GPS times and merge the two windows
def get_triggers(preds_0, preds_5, truncation=0, window_shift=2048,threshold=0.999999, width=2000, delta=256*7):
    
    triggers_0, triggers_5 = {}, {} 
    key = 'detection'
        
    peak_0, left_0, right_0 = find_peaks(preds_0, threshold=threshold, width=[width, 3000], mean=0.95)
    peak_5, left_5, right_5 = find_peaks(preds_5, threshold=threshold, width=[width, 3000], mean=0.95)

    triggers_0[key] = [(x + truncation)/4096 for x in right_0]  
    triggers_5[key] = [(x + truncation + window_shift)/4096 for x in right_5]
    
    # Merge the two windows
    triggers = merge_windows(triggers_0, triggers_5, delta)
    
    return triggers


def close_match(a, b):
    return abs(a - b) < 0.25

def has_close_match_in_all_lists(item, all_lists, considered_items):
    for lst in all_lists:
        if not any(close_match(item, lst_item) for lst_item in lst if lst_item not in considered_items):
            return False
    return True

def get_detections(data_dict):
    all_values = list(data_dict.values())
    detected_items = []
    considered_items = []

    for lst in all_values:
        for item in lst:
            if item not in considered_items and has_close_match_in_all_lists(item, all_values, considered_items):
                detected_items.append(item)
                considered_items.extend([lst_item for other_lst in all_values for lst_item in other_lst if close_match(item, lst_item)])

    return detected_items


def Real_Inference(data_dir, Event):

    Models = ['/root/capsule/data/real_models/real_model_1.h5',
              '/root/capsule/data/real_models/real_model_2.h5',
              '/root/capsule/data/real_models/real_model_3.h5',
              '/root/capsule/data/real_models/real_model_4.h5',
              '/root/capsule/data/real_models/real_model_5.h5',
              '/root/capsule/data/real_models/real_model_6.h5'
              ]

    det_locs = {}
    for i in range(len(Models)):
        Model     = keras.models.load_model(Models[i], custom_objects={'LAMB': LAMB})
        threshold = 0.999999
        width     = 2000

        # Load strains
        fp = h5py.File(data_dir, 'r')
        strain_L1 = fp['strain_L1'][:]
        strain_H1 = fp['strain_H1'][:]
        strain_V1 = fp['strain_V1'][:]

        # Make preds
        preds_0, preds_5, delta = make_preds(strain_L1, strain_H1, strain_V1, Model)

        # Post Process and find triggers
        triggers = get_triggers(preds_0, preds_5, truncation=0, window_shift=2048, threshold=threshold, width=width, delta=delta)
        det_locs[data_dir.split('/')[-1].split('_')[0]+f'_model_{i+1}'] = triggers['detection']
        print(data_dir.split('/')[-1].split('_')[0]+f' model_{i+1}', triggers)


    detection = get_detections(det_locs)
    
    print('Ensemble Detection Results:', detection)
    
    # Calculate padding
    total_length = 4096*4096
    padding_each_side = (total_length - len(strain_L1)) // 2

    # Pad the array
    padded_L1 = np.pad(strain_L1, (padding_each_side, padding_each_side), 'constant', constant_values=(0, 0))
    padded_H1 = np.pad(strain_H1, (padding_each_side, padding_each_side), 'constant', constant_values=(0, 0))
    padded_V1 = np.pad(strain_V1, (padding_each_side, padding_each_side), 'constant', constant_values=(0, 0))

    # Plotting
    plt.figure(figsize=(10,5))
    plt.ylim(-15,15)
    plt.plot(np.linspace(0, 4096, total_length), padded_L1, color='red', label='Livingston')
    plt.xlim(0, 4096)
    plt.xlabel(f'4096s around {Event}')
    plt.ylabel('Strain')
    plt.title('Livingston Detector')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.ylim(-15,15)
    plt.plot(np.linspace(0, 4096, total_length), padded_H1, color='green', label='Hanford')
    plt.xlim(0, 4096)
    plt.xlabel(f'4096s around {Event}')
    plt.ylabel('Strain')
    plt.title('Hanford Detector')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.ylim(-15,15)
    plt.plot(np.linspace(0, 4096, total_length), padded_V1, color='blue',label='Virgo')
    plt.xlim(0, 4096)
    plt.xlabel(f'4096s around {Event}')
    plt.ylabel('Strain')
    plt.title('Virgo Detector')
    plt.legend()
    plt.show()

    # Create an array of size 4096 filled with zeros
    data = np.zeros(4096)

    # Plotting
    plt.figure(figsize=(10,5))
    plt.plot(data, color='orange')

    # Add vertical lines at the positions defined by detection_list
    for det in detection:
        plt.axvline(x=(det+padding_each_side/4096.0), ymin=0.333, ymax=1, color='orange')

    plt.ylim(-0.5, 1)
    plt.xlabel(f'4096s around {Event}')
    plt.ylabel('Detection')
    plt.title('Merger Detection')
    plt.show()

