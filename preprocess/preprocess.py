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
        '''
        从原始数据预处理生成一个Preprocessor对象
        :param convert_fn: 处理函数
        :param file_path: 文件位置
        :return: 一个Preprocessor对象
        '''
        preprocessor = Preprocessor()
        preprocessor.samples = convert_fn(file_path)
        return preprocessor

    @staticmethod
    def load(file_path):
        '''
        从已处理过文件导入生成一个Preprocessor对象
        :param file_path: 文件位置
        :return: 一个Preprocessor对象
        '''
        pass

    def save(self, file_path):
        '''
        存储处理过的数据集
        :param file_path: 文件位置
        '''
        pass
