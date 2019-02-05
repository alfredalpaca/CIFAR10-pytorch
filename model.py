import torch
import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):
	def __init__(self):
		super(network, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x))) 
		#conv1 gives 28x28; pool makes it 14x14	
		x = self.pool(F.relu(self.conv2(x)))
		#conv2 gives 10x10; pool makes it 5x5
		x = x.view(-1, 16 * 5 * 5)
		#-1 is for batch size; 16x5x5 because there are 16 channels 
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# class network(nn.Module): #16 layer VGG
# 	def __init__(self):
# 		super(VGG, self).__init__()
# 		self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
# 		self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
# 		self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
# 		self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
# 		self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
# 		self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
# 		self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
# 		self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
# 		self.pool = nn.MaxPool2d(2, 2)
# 		self.fc1 = nn.Linear(7*7*512, 1024)
# 		self.fc2 = nn.Linear(2048, 512)
# 		self.fc3 = nn.Linear(512, 10)

# 	def forward(self, x):
# 		x = F.relu(self.conv1(x))
# 		x = F.relu(self.conv2(x))
# 		x = self.pool(x)
# 		x = F.relu(self.conv3(x))
# 		x = F.relu(self.conv4(x))
# 		x = self.pool(x)
# 		x = F.relu(self.conv5(x))
# 		x = F.relu(self.conv6(x))
# 		x = F.relu(self.conv6(x))
# 		x = self.pool(x)
# 		x = F.relu(self.conv8(x))
# 		x = F.relu(self.conv9(x))
# 		x = F.relu(self.conv9(x))
# 		x = self.pool(x)
# 		x = F.relu(self.conv9(x))
# 		x = F.relu(self.conv9(x))
# 		x = F.relu(self.conv9(x))
# 		x = self.pool(x)
# 		x = x.view(-1, 7*7*512)
# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))
# 		x = self.fc3(x)
# 		return x
