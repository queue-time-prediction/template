import os, pandas, math
from preprocess.preprocess import Sample, Preprocessor
from utils.time_convert import to_timestamp

def theta_convert_fn(data_path):
    '''
    将csv文件转换为sample数组
    :param data_path: 待处理的csv文件位置
    :return: 一个处理过的sample数组
    '''

    wait_time = [0 for _ in range(30)]
    run_time = [0 for _ in range(30)]

    samples = []
    dataset = pandas.read_csv(data_path)
    for i in dataset.iterrows():
        sample = Sample()
        sample.id = i[0]
        sample.queued_time = to_timestamp(i[1]['QUEUED_TIMESTAMP'])
        sample.start_time = to_timestamp(i[1]['START_TIMESTAMP'])
        sample.end_time = to_timestamp(i[1]['END_TIMESTAMP'])
        sample.requested_quarter = int(float(i[1]['REQUESTED_CORE_HOURS']) * 4)
        sample.nodes_used = int(i[1]['NODES_USED'])

        sample.wait_quarter = (sample.start_time - sample.queued_time) // 900
        sample.run_quarter = (sample.end_time - sample.start_time) // 900

        wait_time[int(math.log2(sample.nodes_used+1))] += 1
        run_time[int(math.log2((sample.requested_quarter / sample.nodes_used) + 1))] += 1

        samples.append(sample)

    samples.sort(key=lambda x:x.start_time)
    return samples



def get_task_list(sorted_list, timestamp):
    pass






def preprocessor_raw() -> Preprocessor:
    return Preprocessor.from_raw(theta_convert_fn, os.path.dirname(os.path.abspath(__file__)) + '/theta.csv')

def preprocessor_load() -> Preprocessor:
    return Preprocessor.load(os.path.dirname(os.path.abspath(__file__)) + '/theta_saved.txt')
