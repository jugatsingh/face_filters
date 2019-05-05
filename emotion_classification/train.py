from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import cv2
import argparse
from model import Net
import time


if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

batch_size = 64
learning_rate = 1e-3
num_epochs = 20
model_dir = 'saved_models'
model_name = 'best_emotion_model.pt'

class Fer2013Dataset(Dataset):
	def __init__(self, path: str):

		#path to dataset x is nxd y is nx1
		with np.load(path) as data:
			self.__samples = data['X']
			self.__labels = data['Y']
		self.__samples = self.__samples.reshape((-1, 1, 48, 48)) #BxCxWxH
		self.X = Variable(torch.from_numpy(self.__samples)).float().to(device)
		self.Y = Variable(torch.from_numpy(self.__labels)).float().to(device)

	def __len__(self):
		return len(self.__labels)

	def __getitem__(self, idx):
		return {'image': self.__samples[idx], 'label': self.__labels[idx]}

trainset = Fer2013Dataset('data/fer2013_train.npz')
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)

testset = Fer2013Dataset('data/fer2013_test.npz')
testloader = DataLoader(testset, batch_size = batch_size, shuffle=False)


def evaluate(outputs, labels):
	Y = labels.data.cpu().numpy()
	Y_pred = np.argmax(outputs.data.cpu().numpy(), axis=1)
	return float(np.sum(Y_pred == Y))

def batch_evaluate(net, dataset, batch_size = 512):
	acc = 0.0
	n = dataset.X.shape[0]
	for i in range(0, n, batch_size):
		x = dataset.X[i: min(i+batch_size, n)]
		y = dataset.Y[i: min(i+batch_size, n)]
		acc += evaluate(net(x), y)
	return acc / n


best_test_acc = -np.inf
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [])

start_time = time.time()
for epoch in range(num_epochs):
	running_loss = 0.0
	for i, data in enumerate(trainloader):
		optimizer.zero_grad()

		inputs = Variable(data['image'].float()).to(device)
		labels = Variable(data['label'].long()).to(device)
		output = net(inputs)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		if i % 100 == 0:
			train_acc = batch_evaluate(net, trainset)
			test_acc = batch_evaluate(net, testset)
			print('Epoch: %d, Batch: %5d, Loss: %.3f, train acc: %.3f, test_acc: %.3f' % (epoch, i, running_loss/(i+1), train_acc, test_acc))
			print('Total Time Elapsed: ', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
			if test_acc > best_test_acc:
				best_test_acc = test_acc
				torch.save(net.state_dict(), model_dir + model_name)
				print('Saving ' + model_dir + model_name)
