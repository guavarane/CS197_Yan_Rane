import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pyspark import SparkContext, SparkConf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

conf = SparkConf().setAppName("MySparkApp")
conf.set("spark.executor.memory", "8g")
sc = SparkContext(conf=conf)

file_path = '/Users/gauravrane/Documents/Spring_2023/CS197/CS197_dataset/Master_bit.csv'
rdd = sc.textFile(file_path)


def divide_inp(line):
    str_inp = line.split(',')[0:2304]
    float_inp = [float(x) for x in str_inp]
    return float_inp


def divide_out(line):
    float_out = float(line.split(',')[2304])
    return float_out


inputs = rdd.map(lambda line: divide_inp(line))
outputs = rdd.map(lambda line: divide_out(line))

print("Length of inputs_tensor_rdd:", inputs.count())


inputs_tensor_rdd = inputs.map(lambda arr: torch.tensor(arr))
outputs_tensor_rdd = outputs.map(lambda arr: torch.tensor(arr))

# Combine inputs and outputs into a single RDD of (input, output) tuples
data_rdd = inputs_tensor_rdd.zip(outputs_tensor_rdd)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.initialLayer = nn.Linear(2304, 512)
        self.secondLayer = nn.Linear(512, 256)
        self.thirdLayer = nn.Linear(256, 64)
        self.finalLayer = nn.Linear(64, 1)
        self.leakyrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        initial = self.initialLayer(x)
        initial_relu = self.leakyrelu(initial)
        second = self.secondLayer(initial_relu)
        second_relu = self.leakyrelu(second)
        third = self.thirdLayer(second_relu)
        third_relu = self.leakyrelu(third)
        final = self.finalLayer(third_relu)
        return final


model = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 32

print('About to start the training loop')
for epoch in range(n_epochs):

    num_samples = 94905
    num_batches = num_samples // batch_size
    batch_rdds = data_rdd.randomSplit([1.0] * num_batches)
    running_loss = 0.0

    for i in range(num_batches):
        selected_batch = batch_rdds[i].collect()
        inputs = [t[0] for t in selected_batch]
        targets = [t[1] for t in selected_batch]
        inputs = [input.to(device) for input in inputs]
        targets = [target.to(device) for target in targets]

        optimizer.zero_grad()
        pred_outputs = model(torch.stack(inputs)).squeeze()
        loss = criterion(pred_outputs, torch.stack(targets))
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss
        print(f"Epoch {epoch + 1}, Batch {i + 1}/{num_batches}, Loss: {batch_loss:.4f}")

    epoch_loss = running_loss / num_batches
    print(f"Epoch {epoch + 1} finished, Average Loss: {epoch_loss:.4f}")