#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains a function to seperate the DNS-Challenge data set in a
training and validation set.

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 13.05.2020

This code is licensed under the terms of the MIT-license.
作用：将数据集划分为训练集（80%）和验证集（20%）
前提是noisy，clean，noise里面文件名相同
"""

import os
from random import seed, sample
from shutil import move


def create_corpus(path, path_new, percent_of_train_material=80):
    '''
    Function to divide a data set.

    Parameters
    ----------
    path : STRING
        Path to dataset directory with three sub directories (noisy,noise,clean).
    path_new : STRING
        Target directory for the divided data set.
    percent_of_train_material : INT   （训练集所占比重）
        Percentage of training material (Default = 80%).

    Returns
    -------
    None.

    '''
    # set seed
    seed(42)
    # create list for folder names. Change if you have other names as the default
    #         mix      noise   speech
    names = ["noisy", "noise", "clean"]
    # create paths to source files
    path_to_noise = os.path.join(path, names[1])  # 路径拼接
    path_to_noisy = os.path.join(path, names[0])
    path_to_clean = os.path.join(path, names[2])
    # create paths to training folders
    path_to_noise_train = os.path.join(path_new, 'train', names[1])
    path_to_noisy_train = os.path.join(path_new, 'train', names[0])
    path_to_clean_train = os.path.join(path_new, 'train', names[2])
    # create paths to validation folders
    path_to_noise_val = os.path.join(path_new, 'val', names[1])
    path_to_noisy_val = os.path.join(path_new, 'val', names[0])
    path_to_clean_val = os.path.join(path_new, 'val', names[2])

    # create directories if not existent
    if not os.path.exists(path_to_noise_train):
        os.makedirs(path_to_noise_train)
    if not os.path.exists(path_to_noisy_train):
        os.makedirs(path_to_noisy_train)
    if not os.path.exists(path_to_noise_val):
        os.makedirs(path_to_noise_val)
    if not os.path.exists(path_to_noisy_val):
        os.makedirs(path_to_noisy_val)
    if not os.path.exists(path_to_clean_val):
        os.makedirs(path_to_clean_val)
    if not os.path.exists(path_to_clean_train):
        os.makedirs(path_to_clean_train)

    # get the filenames
    file_names = os.listdir(path_to_noisy)
    # split file names in training and validation
    train_names = sample(file_names, int(
        percent_of_train_material/100*len(file_names)))  # 从序列a中随机抽取n个元素，并将n个元素生以list形式返回。
    val_names = file_names
    for file in train_names:
        val_names.remove(file)
    # copy files to training folder
    for file in train_names:
        source_file_noisy = os.path.join(path_to_noisy, file)
        source_file_noise = os.path.join(path_to_noise, file)
        source_file_clean = os.path.join(path_to_clean, file)

        dest_file_noisy = os.path.join(path_to_noisy_train, file)
        dest_file_noise = os.path.join(path_to_noise_train, file)
        dest_file_clean = os.path.join(path_to_clean_train, file)

        move(source_file_noisy, dest_file_noisy)
        move(source_file_noise, dest_file_noise)
        move(source_file_clean, dest_file_clean)
    # copy files to validation folder
    for file in val_names:
        source_file_noisy = os.path.join(path_to_noisy, file)
        source_file_noise = os.path.join(path_to_noise, file)
        source_file_clean = os.path.join(path_to_clean, file)

        dest_file_noisy = os.path.join(path_to_noisy_val, file)
        dest_file_noise = os.path.join(path_to_noise_val, file)
        dest_file_clean = os.path.join(path_to_clean_val, file)

        move(source_file_noisy, dest_file_noisy)
        move(source_file_noise, dest_file_noise)
        move(source_file_clean, dest_file_clean)
    # remove empty directories
    os.rmdir(path_to_noisy)
    os.rmdir(path_to_noise)
    os.rmdir(path_to_clean)
    print('Data set divided sucessfully.')


if __name__ == '__main__':
    create_corpus('./training_set', './training_set')
