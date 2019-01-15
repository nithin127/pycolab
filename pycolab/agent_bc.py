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
		self.M = M 							# The number of components

	def forward(self, x):
		self.seg_masks = []
		self.rnn_masks = []
		self.probs = []
		self.latents = []
		inp = self.m(self.f_pre(x))
		seq_len = inp.shape[0]
		prob = torch.zeros(seq_len)
		prob[0] = 1
		self.probs.append(prob)
		self.rnn_masks.append(torch.cumsum(prob, 0))
		for itr in range(self.M):
			x = torch.empty(0,inp[0].shape[0])
			h = torch.zeros(inp[0].shape).reshape(1,1,-1)
			c = torch.zeros(inp[0].shape).reshape(1,1,-1)
			# Masking segments for the b prediction
			for inp_t, mask_t in zip(inp, self.rnn_masks[-1]):
				x_t, (h,c) = self.lstm(inp_t.reshape(1, 1,-1), (h,c))
				x = torch.cat((x, x_t.reshape(1,-1)))
				h = h*mask_t
			# Computing the b
			hb = self.f_b2(F.relu(self.f_b1(x)))
			b = F.gumbel_softmax(hb.transpose(0, 1), 1)
			seg_mask = (self.rnn_masks[-1]*(1 - torch.cumsum(b, 1))).reshape(-1)
			self.seg_masks.append(seg_mask)
			h = torch.zeros(inp[0].shape).reshape(1,1,-1)
			c = torch.zeros(inp[0].shape).reshape(1,1,-1)
			for inp_t, mask_t in zip(inp, seg_mask):
				_, (h,c) = self.lstm(inp_t.reshape(1, 1,-1), (h,c))
				h = h*mask_t
			# Computing the z
			hz = self.f_z2(F.relu(self.f_z1(h.squeeze(1))))
			z = F.gumbel_softmax(hz, 1)
			# Get the masks and append
			self.probs.append(b.squeeze(0))
			self.latents.append(z)
			self.rnn_masks.append(self.rnn_masks[-1]*torch.cumsum(b.squeeze(0),0))
		return self.probs, self.latents, self.seg_masks


class DecoderRNN(nn.Module):
	def __init__(self, n_classes = 9):
		super(DecoderRNN, self).__init__()
		self.n_classes = n_classes
		# layer norm and lstm
		self.m = nn.LayerNorm(20)
		self.lstm = nn.LSTM(20, self.n_classes)
		# deconv into shape = 128
	def forward(self, z, seq_length = 100):
		h = torch.zeros([1,1, self.n_classes])
		c = torch.zeros([1,1, self.n_classes])
		x = torch.empty(0, self.n_classes)
		for _ in range(seq_length):
			x_t, (h,c) = self.lstm(z.reshape(1,1,-1), (h,c))
			x = torch.cat((x, F.softmax(x_t.reshape(1,-1), dim=1)))
		return x


class AgentNetwork:
	def __init__(self):
		# The networks
		self.embed = EmbedCNN()
		self.encoder = EncoderRNN()
		self.decoder = DecoderRNN()
		#Optimisers
		self.optimiser = torch.optim.Adam(list(self.encoder.parameters()) +\
						 list(self.decoder.parameters()) + list(self.embed.parameters()))
		self.criterion = nn.NLLLoss(reduction='none')
		
	def train(self, demos, epochs=100):
		for epoch in range(epochs):
			for demo in demos:
				observations_3d = encode_observation_3d(demo["observations"])
				loss_t = self.train_iter(observations_3d, demo["actions"][:-1])
				print(loss_t)

		return None

	def train_iter(self, observations, actions):
		# We're looking at a single demonstration
		# Embed
		obs = torch.Tensor(observations)
		obs_emb = self.embed(obs)
		# Encode iteratively
		probs, latents, seg_masks = self.encoder(obs_emb)
		# Decode
		trajs = []
		for latent in latents:
			trajs.append(self.decoder(latent, obs_emb.shape[0]))
		# Formulate loss
		loss = torch.zeros([1])
		for tr, mask, b_t, z in zip(trajs, seg_masks, probs, latents):
			# Negative log likelihood of actions
			import ipdb; ipdb.set_trace()
			loss += torch.sum(self.criterion(tr[:-2], torch.tensor(actions))*mask[:-2])/mask[:-2].shape[0]
			# Loss for the priors part
			loss += 0
			b_t + z = ewih
		# Backprop :)
		loss.backward()
		self.optimiser.step()
		return loss.item()

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
	# Create the module and train the agent
	agent = AgentNetwork()
	agent.train(mazes_and_demos["demos"], epochs=100)
	import ipdb; ipdb.set_trace()


if __name__ == '__main__':
	main()


