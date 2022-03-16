from preprocess.theta.theta_convertor import preprocessor_load
from model.linear1.main import data_samples_train


preprocessor = preprocessor_load()
que = [x.queuing for x in data_samples_train]
ans = []
avg = sum(que) / len(que)
for i in que:
    print((i - avg) ** 2)
    ans.append((i - avg) ** 2)

print()
print(sum(ans) / len(ans))