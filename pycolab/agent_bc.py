import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np


class EmbedCNN(nn.Module):
	def __init__(self):
		super(EmbedCNN, self).__init__()
		# Convolutional layers
		self.conv1 = nn.Conv2d(12, 64, (3, 3)) # Size (-1, 12, 10, 30) -> (-1, 64, 8, 28)
		self.conv2 = nn.Conv2d(64, 64, (3, 3)) # Size (-1, 64,  8, 28) -> (-1, 64, 6, 26)
		# FC layers
		self.f1 = nn.Linear(9984, 256) # Input size (-1,  9984)
		
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(x.shape[0], -1)
		return self.f1(x)


class EncoderRNN(nn.Module):
	def __init__(self, K = 20, M = 10):
		super(EncoderRNN, self).__init__()
		# Another Linear Layer + Layer Norm
		self.f_pre = nn.Linear(256, 128)
		self.m = nn.LayerNorm(128)
		# LSTM layer
		self.lstm = nn.LSTM(128,128)
		# MLP layers
		self.f_z1 = nn.Linear(128, 64)
		self.f_z2 = nn.Linear(64, K) 		# The value of K here is 20
		self.f_b1 = nn.Linear(128, 64)
		self.f_b2 = nn.Linear(64, 1)
		# Other stuff
		self.seg_masks = []
		self.rnn_masks = []
		self.probs = []
		self.latents = []
		self.M = M 							# The number of components

	def forward(self, x):
		inp = self.m(self.f_pre(x))
		seq_len = inp.shape[0]
		prob = torch.zeros(seq_len)
		prob[0] = 1
		mask = torch.ones(seq_len)
		self.probs.append(prob)
		self.seg_masks.append(mask)
		self.rnn_masks.append(mask)
		for _ in range(self.M):
			x = torch.empty(0,inp[0].shape[0])
			h = torch.zeros(inp[0].shape).reshape(1,1,-1)
			c = torch.zeros(inp[0].shape).reshape(1,1,-1)
			# Masking segments for the b prediction
			for inp_t, mask_t in zip(inp, self.rnn_masks[-1]):
				x_t, (h,c) = self.lstm(inp_t.reshape(1, 1,-1), (h,c))
				x = torch.cat((x, x_t.reshape(1,-1)))
				h = h*mask_t
			# Masking segments for the z prediction
			h = torch.zeros(inp[0].shape).reshape(1,1,-1)
			c = torch.zeros(inp[0].shape).reshape(1,1,-1)
			for inp_t, mask_t in zip(inp, self.seg_masks[-1]):
				_, (h,c) = self.lstm(inp_t.reshape(1, 1,-1), (h,c))
				h = h*mask_t
			# Computing the hz, hb
			hz = self.f_z2(F.relu(self.f_z1(h.squeeze(1))))
			hb = self.f_b2(F.relu(self.f_b1(x)))
			# Gumbel softmax-ing
			z = F.gumbel_softmax(hz, 1)
			b = F.gumbel_softmax(hb.transpose(0, 1), 1)
			# Get the masks and append
			mask = torch.ones(b.shape)
			for pr in self.probs:
				mask *= torch.cumsum(pr, 0)
			self.rnn_masks.append(mask)
			self.seg_masks.append(mask*(1 - torch.cumsum(b, 1)))
			self.probs.append(b)
			self.latents.append(z)

		return self.probs, self.latents, self.seg_masks


class AgentNetwork:
	def __init__(self, action_list):
		self.action_list = action_list
		self.embed = EmbedCNN()
		self.encoder = EncoderRNN()
		# self.optimiser = torch.optim.Adam(self.encoder.parameters())
		
	def train(self, observations):
		# We're looking at a single demonstration
		# Embed
		obs = torch.Tensor(observations)
		obs_emb = self.embed(obs)
		# Encode iteratively
		# Decode
		# Formulate loss
		# Backprop :)
		return None

	def save(self, observation):
		return None
		
	def load(self):		
		return None


def encode_observation_3d(observations, observation_order='$H&@J/*c!d#P'):
	all_list = []
	for obs in observations:
		all_list.append(np.array([obs.layers[key].astype(int) for key in observation_order]))
	return all_list


def main():
	# You don't really need mazes, it can just be obtained from the demonstrations
	mazes_and_demos = pickle.load(open("mazes_and_demos.pk", "rb"))
	action_list = [0, 1, 2, 3, 5, 6, 7, 8]
	# Create the module to train the net
	agent = AgentNetwork(action_list)
	# Just playing with a single demonstration
	demo = mazes_and_demos["demos"][0]
	observations_3d = encode_observation_3d(demo["observations"])
	agent.train(observations_3d)


if __name__ == '__main__':
	main()


