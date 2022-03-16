import torch, math
from preprocess.theta.theta_convertor import preprocessor_load
from preprocess.preprocess import Sample

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DataSample:
    def __init__(self, sample: Sample):
        self.id = sample.id
        # inputs
        self.requesting = math.log2(sample.requested_quarter+1)
        self.nodes = math.log2(sample.nodes_used+1)
        # output
        self.queuing = math.log2(sample.wait_quarter+1)



data_samples_train = [DataSample(x) for x in preprocessor_load().samples][:50000]
data_samples_test = [DataSample(x) for x in preprocessor_load().samples][50001:70000]

class QueueDataset(Dataset):
    def __init__(self, data_samples):
        self.data_samples = data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, id):
        id = id % len(self.data_samples)
        sample = self.data_samples[id]
        target = torch.Tensor([sample.requesting, sample.nodes])
        label = torch.Tensor([sample.queuing])
        return target, label

dataset_train = QueueDataset(data_samples_train)
train_dataloader = DataLoader(dataset_train, batch_size=30, shuffle=True)
dataset_test = QueueDataset(data_samples_test)
test_dataloader = DataLoader(dataset_test, batch_size=30, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(2,5)
        self.l2 = nn.Linear(5,5)
        self.l4 = nn.Linear(5,1)

    def forward(self, x):
        output = x
        output = F.relu(self.l1(output))
        output = F.relu(self.l2(output))
        output = F.relu(self.l4(output))
        return output

model = NeuralNetwork()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

def train(dataloader, model, loss_fn, optimizer):
    for epoch in range(200):
        model.train()
        loss_tot = []
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                loss_tot.append(round(loss, 1))
        print(epoch, '\t', round(sum(loss_tot)), loss_tot)

def test(test_dataloader, model, loss_fn):
    test_loss = 0
    num_batches = len(test_dataloader)
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print('test loss avg', test_loss)

def use(model, sample: Sample):
    data_sample = DataSample(sample)
    with torch.no_grad():
        pred = model(torch.Tensor([data_sample.requesting, data_sample.nodes]))
        value = pred.tolist()[0]
        return (2 ** value) - 1



if __name__ == '__main__':
    # train(train_dataloader, model, loss_fn, optimizer)
    # torch.save(model, './model.pkt')
    # test(test_dataloader, model, loss_fn)
    samples = preprocessor_load().samples
    model = torch.load('./model.pkt')

    sums = [0 for _ in range(5)]
    nums = [0 for _ in range(5)]

    for i in range(50000, 75000):
        predict_q = round(use(model, samples[i]), 1)
        actual_q = samples[i].wait_quarter

        if actual_q <= 4: #0-1
            sums[0] += abs(predict_q - actual_q)
            nums[0] += 1
        elif actual_q <= 12: #1-3
            sums[1] += abs(predict_q - actual_q)
            nums[1] += 1
        elif actual_q <= 24: #3-6
            sums[2] += abs(predict_q - actual_q)
            nums[2] += 1
        elif actual_q <= 48: #6-12
            sums[3] += abs(predict_q - actual_q)
            nums[3] += 1
        elif actual_q <= 96: #12-24
            sums[4] += abs(predict_q - actual_q)
            nums[4] += 1

    avgs = [round(sums[i]/nums[i], 2) for i in range(5)]
    print(avgs)
    # linear 1 baseline
    # Prediction Error (hours) for Actually wait time of [1h, 1-3h, 3-6h, 6-12h, 12-24h]
    # If you run this code, you will see [5.32, 31.23, 39.52, 29.85, 47.56]

