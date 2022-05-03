#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train the DTLN model in default settings. The folders for noisy and
clean files are expected to have the same number of files and the files to 
have the same name. The training procedure always saves the best weights of 
the model into the folder "./models_'runName'/". Also a log file of the 
training progress is written there. To change any parameters go to the 
"DTLN_model.py" file or use "modelTrainer.parameter = XY" in this file.
It is recommended to run the training on a GPU. The setup is optimized for the
DNS-Challenge data set. If you use a custom data set, just play around with
the parameters.

Please change the folder names before starting the training. 

Example call:
    $python run_training.py

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 13.05.2020

This code is licensed under the terms of the MIT-license.
"""

from DTLNaec_model import DTLN_aec_model
import os

# use the GPU with idx 0 使用idx=0的这块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# activate this for some reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# path to folder containing the noisy or mixed audio training files
path_to_train_mic = '/root/sfj/AECdata/training_set/train/nearend_mic_signal'
path_to_train_lpb = '/root/sfj/AECdata/training_set/train/farend_speech'
# path to folder containing the clean/speech files for training
path_to_train_s1 = '/root/sfj/AECdata/training_set/train/nearend_speech'
# path to folder containing the noisy or mixed audio validation data
path_to_val_mic = '/root/sfj/AECdata/training_set/val/nearend_mic_signal'
path_to_val_lpb = '/root/sfj/AECdata/training_set/val/farend_speech'
# path to folder containing the clean audio validation data
path_to_val_s1 = '/root/sfj/AECdata/training_set/val/nearend_speech'

# name your training run
runName = 'DTLN_aec_model'
# create instance of the DTLN model class
modelTrainer = DTLN_aec_model()
# build the model

modelTrainer.build_DTLN_aec_model()
# modelTrainer.build_CRN_aec_model()

# compile it with optimizer and cost function for training
modelTrainer.compile_model()
# train the model
modelTrainer.train_model(runName, path_to_train_mic, path_to_train_lpb,
                         path_to_train_s1, path_to_val_mic, path_to_val_lpb, path_to_val_s1)
