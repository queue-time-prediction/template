import json

# 样本格式
class Sample:
    def __init__(self):
        self.id = 0
        self.queued_time = 0
        self.start_time = 0
        self.end_time = 0
        self.nodes_used = 0
        self.requested_quarter = 0
        self.wait_quarter = 0
        self.run_quarter = 0

    def __str__(self):
        return f'sample {self.id}'



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
        preprocessor = Preprocessor()
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                s = Sample()
                line_arr = [int(x) for x in line.split(' ')]
                if len(line_arr) == 8:
                    s.id, s.queued_time, s.start_time, s.end_time, s.nodes_used, \
                    s.requested_quarter, s.wait_quarter, s.run_quarter = line_arr
                    preprocessor.samples.append(s)
        return preprocessor




    def save(self, file_path):
        '''
        存储处理过的数据集
        :param file_path: 文件位置
        '''
        with open(file_path, encoding='utf-8', mode='w') as f:
            for s in self.samples:
                f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(s.id, s.queued_time, s.start_time, s.end_time, s.nodes_used,
                s.requested_quarter, s.wait_quarter, s.run_quarter))
