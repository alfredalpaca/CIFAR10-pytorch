import os
import pickle

class CIFAR10():
	def unpickle(self, file):
		with open(file , 'rb') as f:
			dictionary = pickle.load(f, encoding='bytes')
		return dictionary

	def __init__(self, paths, transform=None):
		super(CIFAR10, self).__init__()
		self.dictionary = []
		for p_ in paths:
			rdir = os.path.abspath(p_)
			self.dictionary.append(self.unpickle(rdir))
			pass

	def __len__(self):
		return len(self.dictionary)*10000

	def __getitem__(self, i):
		image = self.dictionary[int(i/10000)][b'data'][i%9999]
		label = self.dictionary[int(i/10000)][b'labels'][i%9999]
		image = image.reshape(-1,32,32)
		return image, label

