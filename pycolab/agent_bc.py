import torch
import torch.nn as nn
import torch.functional as F

import pickle
import numpy as np


class bcNet(nn.Module):
	def __init__(self, action_list):
		super(AgentNetwork, self).__init__()
		self._action_list = action_list
		# Convolutional layers
		self.conv1 = nn.Conv2d(12, 6, (4, 8)) # Size (-1, 12, 10, 30) -> (-1, 5, 8, 28)
		self.conv2 = nn.Conv2d( 6, 3, (4, 8)) # Size (-1,  5,  8, 28) -> (-1, 3, 4, 16)
		# FC layers
		self.f1 = nn.Linear(192, 64) # Input size (-1,  5,  8, 28)
		self.f2 = nn.Linear( 64, 32) # Input size (-1,  5,  8, 28)
		self.f3 = nn.Linear( 32, len(self._action_list)) # Input size (-1,  5,  8, 28)


	def forward(self, observations, actions):
		


class AgentNetwork:		
	def train(self, observation, action_list, batch_size = 16):
		# Batch size per loop


	def save(self, observation):
		

	def load(self):
		


def encode_observation_3d(observations, observation_order='$H&@J/*c!d#P'):
	all_list = []
		for obs in observations:
			all_list.append(np.array([obs.layers[key].astype(int) for key in observation_order]))
	return all_list


def main():
	# You don't really need mazes, it can just be obtained from the demonstrations
	mazes_and_demos = pickle.load(open("mazes_and_demos.pk", "rb"))
	observations = mazes_and_demos["observations"]
	actions = mazes_and_demos["actions"]

	# create_bc_policy(observations, actions)
	agent = AgentNetwork()
	obs_3d = encode_observation_3d(observations)

	agent.train(obs_3d, actions)


if __name__ == '__main__':
	main()


