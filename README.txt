(语音降噪网络,回声消除网络,以及生成DNS数据集)

-(双信号变换LSTM网络)DTLN训练步骤
paper：Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression
ACOUSTIC ECHO CANCELLATION WITH THE DUAL-SIGNAL TRANSFORMATION LSTM NETWORK

安装训练环境：$： conda env create -f train_env.yml

DNS数据集准备：
(注意：数据集各对应音频在文件夹中的文件名必须一致)
在noisyspeech_synthesizer.cfg中修改存放语料的noise_dir和speech_dir路径
运行：python noisyspeech_synthesizer_multiprocessing.py  得到training_set
划分训练集和验证集：python split_dns_corpus.py

开始训练：
（降噪）
run_training.py中修改路径，DTLN_model.py可以修改相关设置
运行：python run_training.py
模型保存路径：/DTLN-master/models_DTLN_model
（回声消除）
run_DTLNaec_training.py中修改路径，DTLNaec_model.py可以修改相关设置
运行：python run_DTLNaec_training.py
模型保存路径：/DTLN-master/models_DTLNaec_model

测试模型：
（降噪）
创建文件夹：input_path(测试集语音路径)，output_path(处理后的语音路径)
找到模型路径：model.h5(储存模型的H5文件)
运行：python run_evaluation.py -i ./input_path -o ./output_path -m ./path/model.h5
（回声消除）
运行：python run_aec_evaluation.py -i ./in -o ./out_aec -m ./models_DTLN_aec_model/DTLN_aec_model.h5

