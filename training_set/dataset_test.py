from cProfile import label
import os
import fnmatch
from tabnanny import verbose
import tarfile
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, \
    EarlyStopping, ModelCheckpoint
import tensorflow as tf
import soundfile as sf
from wavinfo import WavInfoReader
from random import shuffle, seed
import numpy as np


class Dataloader():
    def __init__(self):
        self.len_of_samples = 3
        self.num_samples = 6
        self.create_tf_data_obj()

        self.input1 = []
        self.input2 = []
        self.output = []

    def gen(self):
        for i in range(self.num_samples):
            in_dat = np.array([0, 1, 2])
            in_dat2 = np.array([1, 1, 1])

            tar_dat = np.array([0, 0, 0])
            # print(in_dat.astype('float32'))
            yield [in_dat.astype('float32'), in_dat2.astype('float32')], tar_dat.astype('float32')

    def create_tf_data_obj(self):
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.gen,
            (tf.float32,  tf.float32),
            output_shapes=(tf.TensorShape(None),
                           tf.TensorShape([self.len_of_samples]))
        )
        # [self.len_of_samples]

    def dataset(self):
        for i in range(self.num_samples):
            in_dat = np.array([0, 1, 2])
            in_dat2 = np.array([1, 1, 1])
            tar_dat = np.array([0, 0, 0])
            in_dat = in_dat.astype('float32')
            tar_dat = tar_dat.astype('float32')
            tar_tf_dat = tf.convert_to_tensor(tar_dat, dtype=tf.float32)
            self.input1.append(in_dat)
            self.input2.append(in_dat2)
            self.output.append(tar_tf_dat)

        self.input1 = tf.convert_to_tensor(self.input1, dtype=tf.float32)
        return [self.input1, self.input2, self.output]


# 验证模型

if __name__ == "__main__":

    #  验证soundfile输出类型为numpy，并将numpy转化成Tensor
    # audio1, fs = sf.read('1.wav')
    # audio2, fs2 = sf.read('1.wav')

    # audio = [audio1.astype('float32'), audio2.astype('float32')]
    # print(audio[:3])
    # print('!!!audio type:', type(audio))
    # audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    # print(audio_tf[:3])
    # print('!!!audio_tf type:', type(audio_tf))

    # 验证Dataloader功能
    gen_input = Dataloader()
    dataset = gen_input.tf_data_set
    print('!!! element_spec:', dataset.element_spec)
    print('!!! list: ', list(dataset.as_numpy_iterator()))
    print('!!!dataset:', dataset)

    dataset = gen_input.dataset()
    input1 = dataset[0]
    print("!!!!!!!input1:", input1)
    tar = dataset[2]
    print("!!!!!!!tar:", tar)
    # dataset = dataset.batch(2)
    # print('!!!!batch_list:', list(dataset.as_numpy_iterator()))

    # for batch_idx, (mic_dat, lpb_dat, tar_dat) in enumerate(dataset):
    #     print('batch_idx:', batch_idx)
    #     print('mic_dat:', mic_dat)
    #     print('lpb_dat:', lpb_dat)
    #     print('tar_dat:', tar_dat)

    # DTLNaec_model
    # model = DTLN_model()
    # model.build_DTLN_aec_model()
