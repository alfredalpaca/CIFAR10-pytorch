import os
from dataloader import CIFAR10
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datautil
from model import network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

print(device)

paths = ['./data/cifar-10-batches-py/data_batch_1',
		 './data/cifar-10-batches-py/data_batch_2',
		 './data/cifar-10-batches-py/data_batch_3',
		 './data/cifar-10-batches-py/data_batch_4',
		 './data/cifar-10-batches-py/data_batch_5']

def main():
	data = CIFAR10(paths)
	data_loader = datautil.DataLoader(dataset=data, batch_size=512, num_workers=8, shuffle=True)
	epoch = 0
	net = network().to(device)
	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	running_loss = 0
	for ep in tqdm(range(epoch, 50)):
		for i, (image,label) in tqdm(enumerate(data_loader)):
			optimizer.zero_grad()
			# print (repr(image))
			output = net(image.to(device,dtype=torch.float32))
			# print(label)
			loss = criterion(output, label.to(device))
			loss.backward()
			optimizer.step()
			# running_loss += loss.item()
			if i%2000 == 0:
				print(loss)
			pass
	torch.save(net.state_dict(), './model/model1.pt')


if __name__ == '__main__':
	main()