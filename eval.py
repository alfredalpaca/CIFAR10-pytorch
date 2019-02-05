import os
from dataloader import CIFAR10
import torch
import torch.nn as nn
import torch.utils.data as datautil
from model import network
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

print(device)

paths = ['/home/alfred_alpaca/Code/torch/cifar10/data/cifar-10-batches-py/test_batch']

def main():
	data = CIFAR10(paths)
	data_loader = datautil.DataLoader(dataset=data, batch_size=512, num_workers=8, shuffle=False)
	net = network().to(device)
	net.load_state_dict(torch.load('/home/alfred_alpaca/Code/torch/cifar10/model/model1.pt'))
	net.eval()
	criterion = nn.CrossEntropyLoss().to(device)
	for i, (image,label) in tqdm(enumerate(data_loader)):
		output = net(image.to(device,dtype=torch.float32))
		loss = criterion(output, label.to(device))
		if i%200 == 0:
			print(loss)
		pass
	print(loss)

if __name__ == '__main__':
	main()