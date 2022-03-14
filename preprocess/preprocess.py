import json

# 样本格式
class Sample:
    def __init__(self):
        '''
        这里填Sample具有的参数
        '''
        pass

    def __str__(self):
        pass



# 预处理器
class Preprocessor:
    def __init__(self):
        self.samples = []

    @staticmethod
    def from_raw(convert_fn, file_path):
        preprocessor = Preprocessor()
        preprocessor.samples = convert_fn(file_path)
        preprocessor.save()
        return preprocessor

    @staticmethod
    def load(file_path):
        pass

    def save(self):
        pass
