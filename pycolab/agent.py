import torch
import torch.nn as nn
import torch.functional as F

import numpy as np

OBS_ORDER = '$H&@J/*c!d#P'
TARGET_SEQUENCE = ['v$']
np.random.shuffle(TARGET_SEQUENCE)

class AgentNetwork:
	def __init__(self, observation_order = OBS_ORDER, target_sequence = TARGET_SEQUENCE):
		self._observation_order = observation_order
		self._target_sequence = target_sequence
		self._dict_direction_pickup = {(-1, 0): 6, (1, 0): 5, (0, 1): 7, (0, -1): 8}
		self._dict_direction_move = {(-1, 0): 0, (1, 0): 1, (0, 1): 3, (0, -1): 2}

		# The planning part
		self._current_target =  None
		self._final_move = None
		self._current_plan = None 				# Optional, if we want to plan the whole thing together
		self._plan_index = 0
		self._current_position = None
		self._empty_space = None
		self._current_observation_3d = None


	def encode_observation_3d(self, obs):
		if not self._observation_order:
			self._observation_order = list(obs.keys())
		self._current_observation_3d = np.array([obs[key].astype(int) for key in self._observation_order])


	def select_adjacent(self, i, j):
		check_list = [(-1, 0), (1, 0), (0, 1), (0, -1)]
		np.random.shuffle(check_list)
		for di, dj in check_list:
			if self._empty_space[i + di, j + dj] == 1:
				self._final_move = self._dict_direction_pickup[(di, dj)]
				return (i+di, j+dj)		
		return None


	def neighbors(self, pos):
		i, j = pos
		neighbors = []

		if i > 0:
			if self._empty_space[i-1, j]:
				neighbors.append((i-1, j, (-1, 0)))
		if j > 0:
			if self._empty_space[i, j-1]:
				neighbors.append((i, j-1, (0, -1)))
		if i < self.ij_limit[0] - 1:
			if self._empty_space[i+1, j]:
				neighbors.append((i+1, j, (1, 0)))
		if j < self.ij_limit[1] - 1:
			if self._empty_space[i, j+1]:
				neighbors.append((i, j+1, (0, 1)))

		np.random.shuffle(neighbors)
		return neighbors


	def set_current_plan(self, start):
		"""
		Dijstra's implementation
		"""		
		cost_map = np.inf*np.ones(self._empty_space.shape)
		dir_map = [[(0,0) for i in range(self._empty_space.shape[1])] for j in range(self._empty_space.shape[0])]
		cost_map[start[0], start[1]] = 0
		to_visit = []
		to_visit.append(start)
		while len(to_visit) > 0:
			curr = to_visit.pop(0)
			for nx, ny, d in self.neighbors(curr):
				cost = cost_map[curr[0],curr[1]] + 1
				if cost < cost_map[nx,ny]:
					to_visit.append((nx, ny))
					dir_map[nx[0]][ny[0]] = d
					cost_map[nx,ny] = cost

		seq = []
		curr = self._current_target
		d_curr = dir_map[curr[0]][curr[1]]
		curr = (curr[0] - d_curr[0], curr[1] - d_curr[1])
		seq.append(self._dict_direction_move[d_curr])
		while not curr == start:
			d_curr = dir_map[curr[0]][curr[1]]
			curr = (curr[0] - d_curr[0], curr[1] - d_curr[1])
			seq.append(self._dict_direction_move[d_curr])
		seq.reverse()
		if self._final_move:
			self._current_plan = seq + [self._final_move]
		else:
			self._current_plan = seq


	def set_empty_space(self, observation):
		self._empty_space = np.ones(observation.board.shape)
		for char in OBS_ORDER:
			self._empty_space -= observation.layers[char]


	def next_iter(self, observation):
		self.encode_observation_3d(observation.layers)

		# Deciding the target
		arg, char = self._target_sequence[self._current_index]
		tar_i, tar_j = np.where(observation.layers[char])
		index = np.random.choice(len(tar_i))
		self.set_empty_space(observation)
		# Are we visiting or picking ?
		if arg == 'v':
			self._current_target = (tar_i[index], tar_j[index])
			self._empty_space[tar_i[index], tar_j[index]] = 1
		elif arg == 'p':
			self._current_target = self.select_adjacent(tar_i[index], tar_j[index])
		
		self._plan_index = 0
		self.set_current_plan(np.where(observation.layers['P']))


	def agent_network(self, observation, action_list):
		del action_list # This is only needed for a neural network; and mapping later on

		# If this is the first time calling this function
		if not self._current_plan:
			self._current_index = 0
			self.ij_limit = observation.board.shape 
			self.next_iter(observation)

		self._plan_index += 1
		if self._plan_index > len(self._current_plan):
			self._current_index += 1
			if len(self._target_sequence) == self._current_index:
				print("all the instructions successfully executed")
				return 9, None
			else:
				self.next_iter(observation)
			# reset_to_next_target
		return self._current_plan[self._plan_index - 1], self._target_sequence[self._current_index]
	
