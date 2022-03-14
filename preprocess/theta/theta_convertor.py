import os, pandas
from preprocess.preprocess import Sample, Preprocessor

def theta_convert_fn(data_path):
    '''
    将csv文件转换为sample数组
    :param data_path: 待处理的csv文件位置
    :return: 一个处理过的sample数组
    '''
    pass












def preprocessor_raw() -> Preprocessor:
    return Preprocessor.from_raw(theta_convert_fn, os.path.dirname(os.path.abspath(__file__)) + '/theta.csv')

def preprocessor_load() -> Preprocessor:
    return Preprocessor.load(os.path.dirname(os.path.abspath(__file__)) + '/theta_saved.txt')
