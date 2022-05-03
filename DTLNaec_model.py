# -*- coding: utf-8 -*-
"""
2022/3 Su fengjie
DTLNaec_model.py 中可选DTLN_aec和CRN_aec两个网络 用于处理AEC任务

This File contains everything to train the DTLNaec_model.

For running the training see "run_training.py".
To run evaluation with the provided pretrained model see "run_evaluation.py".

"""
import os
import fnmatch
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


class audio_generator():
    '''
    Class to create a Tensorflow dataset based on an iterator from a large scale
    audio dataset. This audio generator only supports single channel audio files.
    '''

    def __init__(self, path_to_mic, path_to_lpb, path_to_s1, len_of_samples, fs, train_flag=False):
        '''
                Constructor of the audio generator class.
                Inputs:
                    path_to_input       path to the mixtures
                    path_to_s1          path to the target source data
                    len_of_samples      length of audio snippets in samples  分段长度，被设置为5s长度的采样点数
                    fs                  sampling rate
                    train_flag          flag for activate shuffling of files  打乱数据
        '''
        # 设置输入的属性
        self.path_to_mic = path_to_mic  # 麦克风采集信号路径
        self.path_to_lpb = path_to_lpb  # 远端参考信号路径
        self.path_to_s1 = path_to_s1  # 近端语音路径
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.train_flag = train_flag
        # 计算样本数
        self.count_samples()
        # 创建可迭代的tf.data.Dataset对象
        # self.create_tf_data_obj()

        # 创建数据集
        self.input_mic = []
        self.input_lpb = []
        self.lable = []

    def count_samples(self):
        '''
        计算样本总数量(所有音频切成多少小段)小段:len_of_samples为15秒的数据
        '''
        # # path里的文件放入列表，然后筛选出.wav文件
        self.file_names = fnmatch.filter(os.listdir(self.path_to_mic), '*.wav')
        self.total_samples = 0
        # print("file_name!!!!!!!!!!!!!!!!!!:", self.file_names)
        for file in self.file_names:
            info = WavInfoReader(os.path.join(self.path_to_mic, file))
            self.total_samples = self.total_samples + \
                int(np.fix(info.data.frame_count/self.len_of_samples))

    # def create_generator(self):
    #     '''
    #     切片数据集里的所有数据,每个音频分成num_sample个小段输出,输出每一帧
    #     需要保证数据集中mic,lpb,clean文件夹中的文件命名一致且对应
    #     file_process用于处理文件名,split_AECdata_corpus.py用于划分数据集
    #     '''
    #     # train_flag用于区分是训练或测试
    #     # if self.train_flag:
    #     #     shuffle(self.file_name) # 随机排序

    #     # 迭代所有文件 （此处的帧len_of_samples，代表输入网络的小段音频）
    #     for file in self.file_names:
    #         # 读取音频文件(注：三个文件夹内的各个音频文件名对应且相同)
    #         mic_speech, fs_1 = sf.read(os.path.join(self.path_to_mic, file))
    #         lpb_speech, fs_2 = sf.read(os.path.join(self.path_to_lpb), file)
    #         speech, fs_3 = sf.read(os.path.join(self.path_to_s1, file))
    #         # 检查采样率是否满足要求
    #         if fs_1 != self.fs or fs_3 != self.fs:
    #             raise ValueError('Sample rates do not match.')
    #         if mic_speech.ndim != 1 or speech.ndim != 1:
    #             raise ValueError('Too many audio channels. The DTLN audio_generator \
    #                              only supports single channel audio data.')
    #         # 计算一个音频文件的帧数（小段数）
    #         num_samples = int(np.fix(mic_speech.shape[0]/self.len_of_samples))
    #         # 迭代该文件的所有帧
    #         for idx in range(num_samples):
    #             in_dat_mic = mic_speech[int(idx*self.len_of_samples):int((idx+1) *
    #                                                                      self.len_of_samples)]
    #             in_dat_lpb = lpb_speech[int(idx*self.len_of_samples):int((idx+1) *
    #                                                                      self.len_of_samples)]
    #             tar_dat = speech[int(idx*self.len_of_samples):int((idx+1) *
    #                                                               self.len_of_samples)]
    #             yield [in_dat_mic.astype('float32'), in_dat_lpb.astype('float32')], tar_dat.astype('float32')

    def creat_dataset(self):

        for file in self.file_names:
            # 读取音频文件(注：三个文件夹内的各个音频文件名对应且相同)
            mic_speech, fs_1 = sf.read(os.path.join(self.path_to_mic, file))
            lpb_speech, fs_2 = sf.read(os.path.join(self.path_to_lpb, file))
            speech, fs_3 = sf.read(os.path.join(self.path_to_s1, file))
            # 检查采样率是否满足要求
            if fs_1 != self.fs or fs_3 != self.fs:
                raise ValueError('Sample rates do not match.')
            if mic_speech.ndim != 1 or speech.ndim != 1:
                raise ValueError('Too many audio channels. The DTLN audio_generator \
                                 only supports single channel audio data.')
            # 计算一个音频文件的帧数（小段数）
            num_samples = int(np.fix(mic_speech.shape[0]/self.len_of_samples))
            # 迭代该文件的所有帧（小段） 小段长为len_of_samples
            for idx in range(num_samples):
                in_dat_mic = mic_speech[int(idx*self.len_of_samples):int((idx+1) *
                                                                         self.len_of_samples)]
                in_dat_lpb = lpb_speech[int(idx*self.len_of_samples):int((idx+1) *
                                                                         self.len_of_samples)]
                tar_dat = speech[int(idx*self.len_of_samples):int((idx+1) *
                                                                  self.len_of_samples)]
                #
                self.input_mic.append(in_dat_mic.astype('float32'))
                self.input_lpb.append(in_dat_lpb.astype('float32'))
                self.lable.append(tar_dat.astype('float32'))
        self.input_mic = tf.convert_to_tensor(self.input_mic, dtype=tf.float32)
        self.input_lpb = tf.convert_to_tensor(self.input_lpb, dtype=tf.float32)
        self.lable = tf.convert_to_tensor(self.lable, dtype=tf.float32)
        return [self.input_mic, self.input_lpb, self.lable]

    # def create_tf_data_obj(self):
    #     '''
    #     生成tf数据集
    #     传入的是一个generator,即返回字段为yield的函数,不可传入嵌套生成器
    #     '''
    #     self.tf_data_set = tf.data.Dataset.from_generator(
    #         self.create_generator,
    #         (tf.float32, tf.float32),
    #         output_shapes=(tf.TensorShape(None),
    #                        tf.TensorShape([self.len_of_samples])),
    #         args=None)


class DTLN_aec_model():
    '''
    该类用于创建和训练 DTLN model 包含了网络搭建，编译，训练代码
    '''

    def __init__(self):
        # 定义默认的损失函数

        self.cost_function = self.snr_cost
        # 定义空model
        self.model = []
        # 定义默认参数
        self.fs = 16000
        self.batchsize = 8
        self.len_samples = 5  # 秒数，在train_model()中用于确定len_of_samples的长度
        self.activation = 'sigmoid'
        self.numUnits = 128  # LSTM层神经元个数 128
        self.numLayer = 2       # LSTM层数
        self.blockLen = 512     # 帧长
        self.block_shift = 128  # 帧移
        self.dropout = 0.25
        self.lr = 1e-3
        self.max_epochs = 70
        self.encoder_size = 256
        self.eps = 1e-7

        # 定义build_CRN_aec_model的参数
        self.CRN_encoder_size = 320
        self.lstm_num = 4
        self.block_len = 512  # 帧长
        self.frame_shift = 128  # 帧移

        # str()函数将指定的值转换为字符串。
        # 为获得可复现的结果设置环境变量为 42 （随机生成的种子值是不变的）
        os.environ['PYTHONHASHSEED'] = str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        # 在TF 2.x中正确地找到一些库,获得当前主机上某种特定运算设备类型(如gpu)的列表
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)

    @ staticmethod
    def snr_cost(s_estimate, s_true):
        '''
        训练目标loss:计算负信噪比(对最后的维度计算）  维度：(batchsize, len_in_samples)
        '''
        # tf.reduce.mean 计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，降维
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
            (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True)+1e-7)
        num = tf.math.log(snr)
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10*(num / (denom))  # 换底公式
        return loss

    def lossWrapper(self):
        '''
        封装loss函数的装饰器,修改self.cost_function可以更换loss函数
        '''
        def lossFunction(y_true, y_pred):
            # 计算loss并且降维
            loss = tf.squeeze(self.cost_function(y_pred, y_true))
            # 计算batch的loss平均值
            loss = tf.reduce_mean(loss)
            return loss
        return lossFunction

    '''
    以下是一些自定义层的定义
    '''

    def stftLayer(self, x):
        '''
        Lambda层为自定义层,该层是STFT变换,输入时域信号，返回幅值和相位

        '''
        # 分帧操作，返回矩阵[帧数，帧长] 若多通道则为[通道，帧数，帧长]
        # 帧数 num_frames = 1 + (N - frame_size) // frame_step
        frame = tf.signal.frame(x, self.blockLen, self.block_shift)
        # stft 返回 NFFT/2+1 此处NFFT=none 即为frame的帧长
        stft_dat = tf.signal.rfft(frame)
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        return [mag, phase]

    def frame(self, x):
        '''
        CRN网络中 分帧操作
        '''
        frame = tf.signal.frame(x, self.block_len, self.frame_shift)
        return frame

    def ifftLayer(self, x):
        '''
        Lambda层, 作用是逆FFT变换,输入的x为列表[mag, phaes],返回时域帧
        注意:计算的是最内的维度的ISTFT
        '''
        # 计算复数表示 tf.cast数据类型转换，x[0]是幅值 x[1]是相位角
        s1_stft = (tf.cast(x[0], tf.complex64) *
                   tf.exp((1j * tf.cast(x[1], tf.complex64))))
        # 返回时域帧
        return tf.signal.irfft(s1_stft)

    def overlapAddLayer(self, x):
        '''
        Lambda层, 重叠相加,按照帧移frame_step将时域帧重构成时域波形
        输入维度[..., frames, frame_length]，输出[..., output_size]
        output_size = (frames - 1) * frame_step + frame_length
        '''
        return tf.signal.overlap_and_add(x, self.block_shift)

    def CRN_overlapAddLayer(self, x):
        '''
        CRN网络的overlap层
        '''
        return tf.signal.overlap_and_add(x, self.frame_shift)

    def seperation_kernel(self, num_layer, mask_size, x, stateful=False):
        '''
        (注意:该层不能当作Lambda层,否则梯度更新错误)
        部分网络块:2层LSTM+1层dense+1层激活函数sigmoid
        输入:
                x: 对应的特征  (STFT或卷积层输出特征表示)
                num_layer: LSTM层数设定
                mask_size: 输出mask的维度 (即Dense层的维度)
        输出mask
        '''
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True,
                     stateful=stateful)(x)
            if idx < (num_layer-1):
                x = Dropout(self.dropout)(x)
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        return mask

   

    def build_CRN_aec_model(self):
        mic_time_dat = Input(batch_shape=(
            None, None), name='main_input')
        lpb_time_dat = Input(batch_shape=(
            None, None), name='auxiliary_input')
        # 在dim=1增加维度
        mic_frame_dat = Lambda(self.frame, name='mic_frame')(mic_time_dat)
        lpb_frame_dat = Lambda(self.frame, name='lpb_frame')(lpb_time_dat)
        # mic_time_dat_dim = tf.expand_dims(mic_frame_dat, axis=1)
        # lpb_time_dat_dim = tf.expand_dims(lpb_frame_dat, axis=1)
        encoded_feature = Conv1D(self.CRN_encoder_size, 1,
                                 strides=1, use_bias=False, name='MicSignal_encoder')(mic_frame_dat)
        encoded__lpb_feature = Conv1D(self.CRN_encoder_size, 1,
                                      strides=1, use_bias=False, name='LpbSignal_encoder')(lpb_frame_dat)
        encoded_feature_norm = InstantLayerNormalization()(encoded_feature)
        encoded_lpb_feature_norm = InstantLayerNormalization()(encoded__lpb_feature)
        # 拼接两编码器特征
        enocded_frames_norm = keras.layers.concatenate(
            [encoded_feature_norm, encoded_lpb_feature_norm])
        mask = self.seperation_kernel(
            self.lstm_num, self.CRN_encoder_size, enocded_frames_norm)
        estimated = Multiply()([encoded_feature, mask])
        # 解码器
        decoded_frames = Conv1D(
            self.block_len, 1, padding='causal', use_bias=False)(estimated)
        # OLA恢复时域波形
        estimated_sig = Lambda(self.CRN_overlapAddLayer,
                               name='OverlapAdd')(decoded_frames)
        # 创建模型
        self.model = Model(
            inputs=[mic_time_dat, lpb_time_dat], outputs=estimated_sig)
        # 打印模型
        print(self.model.summary())
        # 保存模型图
        # keras.utils.plot_model(self.model, "DTLNaec_model.png")

    def build_DTLN_aec_model(self, norm_stft=False):
        '''
        完整DTLNaec网络搭建
        模型将维度为(batch, len_in_samples)的时域信号进行增强
        输出为:OLA后的语音时域波形
        '''
        time_dat = [Input(batch_shape=(None, None), name='main_input'), Input(batch_shape=(
            None, None), name='auxiliary_input')]
        mic_time_dat = time_dat[0]
        lpb_time_dat = time_dat[1]
        # mic_time_dat = Input(batch_shape=(
        #     None, None), name='main_input')
        # lpb_time_dat = Input(batch_shape=(
        #     None, None), name='auxiliary_input')
        mag_mic, angle_mic = Lambda(
            self.stftLayer, name='mic_stft')(mic_time_dat)
        mag_lpb, angle_lpb = Lambda(
            self.stftLayer, name='lpb_stft')(lpb_time_dat)
        # 是否经过iLN层
        if norm_stft:
            mag_mic_norm = InstantLayerNormalization()(tf.math.log(mag_mic + 1e-7))
            mag_lpb_norm = InstantLayerNormalization()(tf.math.log(mag_lpb + 1e-7))
        else:
            mag_mic_norm = mag_mic
            mag_lpb_norm = mag_lpb
        # 拼接mic和lpb的幅度谱特征
        mag_norm = keras.layers.concatenate([mag_mic_norm, mag_lpb_norm])
        mask_1 = self.seperation_kernel(
            self.numLayer, self.blockLen//2+1, mag_norm)  # blockLen//2+1是输出的size
        # mic幅度谱与mask_1相乘
        estimated_mag = Multiply()([mag_mic, mask_1])
        # 转换回时域
        estimate_frames_1 = Lambda(self.ifftLayer, name='ifft_estimate')(
            [estimated_mag, angle_mic])  # 幅度谱增强后的输出信号
        lpb_frames_1 = Lambda(self.ifftLayer, name='ifft_lpb')(
            [mag_lpb, angle_lpb])
        # 第二部分 编码器
        encoded_feature = Conv1D(self.encoder_size, 1,
                                 strides=1, use_bias=False, name='estimate_encoder')(estimate_frames_1)
        encoded__lpb_feature = Conv1D(self.encoder_size, 1,
                                      strides=1, use_bias=False, name='lpb_encoder')(lpb_frames_1)
        # iLN层
        encoded_feature_norm = InstantLayerNormalization()(encoded_feature)
        encoded_lpb_feature_norm = InstantLayerNormalization()(encoded__lpb_feature)
        # 拼接两编码器特征
        enocded_frames_norm = keras.layers.concatenate(
            [encoded_feature_norm, encoded_lpb_feature_norm])
        mask_2 = self.seperation_kernel(
            self.numLayer, self.encoder_size, enocded_frames_norm)
        estimated = Multiply()([encoded_feature, mask_2])
        # 解码器
        decoded_frames = Conv1D(
            self.blockLen, 1, padding='causal', use_bias=False)(estimated)
        # OLA恢复时域波形
        estimated_sig = Lambda(self.overlapAddLayer,
                               name='OverlapAdd')(decoded_frames)

        # 创建模型
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        # 打印模型
        print(self.model.summary())
        # 保存模型图
        # keras.utils.plot_model(self.model, "DTLNaec_model.png")

    def compile_model(self):
        '''
        编译模型
        '''
        optimizerAdam = keras.optimizers.Adam(lr=self.lr, clipnorm=3.0)
        self.model.compile(loss=self.lossWrapper(), optimizer=optimizerAdam)

    # def creat_saved_model(self, weight_file, target_name):
    #     if weight_file.find('_norm_')!=-1:
    #         norm_stft = True
    #     else:
    #         norm_stft = False
    #     self.build_DTLN_model_stateful(norm_stft=norm_stft)
    #     # load weights
    #     self.model.load_weights(weights_file)
    #     # save model
    #     tf.saved_model.save(self.model, target_name)

    def train_model(self, runName, path_to_train_mic, path_to_train_lpb,
                    path_to_train_s1, path_to_val_mic, path_to_val_lpb, path_to_val_s1):
        '''
        训练 DTLN_aec模型的设置
        回调(callback):ReduceLROnPlateau()用于减小学习率,EarlyStopping()提前结束训练
        monitor:监测的值,可以是accuracy,val_loss,val_accuracy
        factor:缩放学习率的值,学习率将以lr = lr*factor的形式被减少
        patience:当patience个epoch过去而模型性能不提升时,学习率减少的动作会被触发
        mode:'auto','min','max'之一 默认'auto'就行
        epsilon:阈值，用来确定是否进入检测值的“平原区”
        cooldown:学习率减少后,会经过cooldown个epoch才重新进行正常操作
        min_lr:学习率最小值,能缩小到的下限
        '''
        savePath = './models_'+runName+'/'
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # 回调函数：生成文本文件，记录训练的loss
        csv_logger = CSVLogger(savePath + 'train_' +
                               runName + '.log')
        # 回调 ：自动更改减小学习率
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=3, min_lr=10**(-10), cooldown=1)
        # 回调：控制过早停止训练
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                       patience=10, verbose=0, mode='auto', baseline=None)
        # 创建检查点以保存最好的模型
        checkpointer = ModelCheckpoint(savePath+runName+'.h5',
                                       mointor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq='epoch'
                                       )

        # 计算样本中音频块的长度 15秒长的数据（16k*15=240k）
        len_in_samples = int(np.fix(self.fs*self.len_samples/self.block_shift)
                             * self.block_shift)  # np.fix返回 向0取整的浮点数
        # 创建生成器对象，生成训练数据
        generator_input = audio_generator(path_to_train_mic,
                                          path_to_train_lpb,
                                          path_to_train_s1,
                                          len_in_samples,
                                          self.fs, train_flag=True)
        dataset = generator_input.creat_dataset()  # 生成tensorflow类型的数据集

        # calculate number of training steps in one epoch   计算一个epoch训练的样本的步骤数：总数量/batchsize
        steps_train = generator_input.total_samples//self.batchsize
        # create data generator for validation data   生成交叉验证集合
        generator_val = audio_generator(path_to_val_mic,
                                        path_to_val_lpb,
                                        path_to_val_s1,
                                        len_in_samples,
                                        self.fs, train_flag=True)
        dataset_val = generator_val.creat_dataset()
        train_val_x = [dataset_val[0], dataset_val[1]]
        train_val_y = dataset_val[2]
        # calculate number of validation steps
        # validation steps 该参数应<= （验证样本总数/batchsize），每epoch用于验证的样本数为steps_val*batchsize
        steps_val = generator_val.total_samples//self.batchsize  # 该处用整个验证集来验证每个epochs
        # start the training of the model  这里batch_size=None是因为 前面生成数据集时已经划分batchsize了，这里不用再设置，不然还会再分更小的size
        # 使用 fit 方法使模型与训练数据“拟合”
        # callbacks(回调)是传递给模型的对象，用于在训练期间自定义该模型并扩展其行为。如：停止训练，学习率变化，检查点保存，tensorboard可视化等
        self.model.fit(
            x=[dataset[0], dataset[1]],
            y=dataset[2],
            batch_size=self.batchsize,
            steps_per_epoch=steps_train,
            epochs=self.max_epochs,
            verbose=1,
            validation_data=(train_val_x, train_val_y),
            validation_steps=steps_val,
            callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping],
            max_queue_size=10,
            workers=1,
            use_multiprocessing=True)
        tf.keras.backend.clear_session()  # 清除模型占用内存


class InstantLayerNormalization(Layer):
    '''
    瞬时层归一化(iLN)
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        '''
        Method to build the weights.
        '''
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     trainable=True,
                                     name='gamma')
        # initialize beta
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    trainable=True,
                                    name='beta')

    def call(self, inputs):
        '''
        Method to call the Layer. All processing is done here.
        '''

        # calculate mean of each frame
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        # calculate variance of each frame
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean),
                                       axis=[-1], keepdims=True)
        # calculate standard deviation
        std = tf.math.sqrt(variance + self.epsilon)
        # normalize each frame independently
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


if __name__ == '__main__':
    model = DTLN_aec_model()
    model.build_DTLN_aec_model()
    # model.compile_model()
    # model.train()
