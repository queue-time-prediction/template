import os, pandas
from preprocess.preprocess import Sample, Preprocessor

def example_convert_fn(data_path):
    raise NotImplementedError('')


def preprocessor_raw() -> Preprocessor:
    return Preprocessor.from_raw(example_convert_fn, os.path.dirname(os.path.abspath(__file__)) + '/example.csv')

def preprocessor_load() -> Preprocessor:
    return Preprocessor.load(os.path.dirname(os.path.abspath(__file__)) + '/example_saved.txt')
