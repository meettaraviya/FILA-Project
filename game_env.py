from games import *
import numpy as np

from keras.models import Sequential
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, Dense, Maximum
from keras import optimizers
import keras

limit = 4
n_rows = 4

def get_nims_random_move(board):

	move = [None, None]
	move[0] = np.random.choice(np.where(board>0)[0])
	move[1] = np.random.randint(1, 1+board[move[0]])
	return move

def encode(state, action):
	# try:
	state[action[0]] -= action[1]
	ans = np.array(list(map(lambda s: list(map(int, np.binary_repr(s).zfill(limit))), state))).flatten().reshape((1, -1))
	state[action[0]] += action[1]
	return ans
	# except Exception:
	# 	import pdb; pdb.set_trace()

class EpsilonRandomNimAgent:
	def __init__(self, epsilon):
		self.epsilon = epsilon
	def get_move(self, board):

		move = [None, None]
		if np.random.binomial(1, self.epsilon) == 1:
			return get_nims_random_move(board)

		else:
			xor = np.bitwise_xor.reduce(board)
			if xor != 0:
				move = [None, None]
				move[0] = np.where(np.bitwise_xor(board, xor)<board)[0][0]
				move[1] = board[move[0]] - (board[move[0]]^xor)
				return move

			else:
				return get_nims_random_move(board)
	
	def gameOver(self,win):
		pass

class OptimalNimAgent:

	def get_move(self, board):
		xor = np.bitwise_xor.reduce(board)
		if xor != 0:
			move = [None, None]
			move[0] = np.where(np.bitwise_xor(board, xor)<board)[0][0]
			move[1] = board[move[0]] - (board[move[0]]^xor)
			return move

		else:
			return get_nims_random_move(board)
	
	def gameOver(self,win):
		pass

class OptimalNimEnv:

	def __init__(self):

		self.nim_game = NimsGame(np.random.randint(2**limit, size=n_rows))
		self.opponent = OptimalNimAgent()

	def get_state(self):

		return self.nim_game.board

	def play_action(self, action):

		done = False
		prev_state = self.nim_game.board.copy()
		self.nim_game.play_move(action)
		reward = 1 if np.bitwise_xor.reduce(self.nim_game.board) == 0 else 0
		next_state = self.nim_game.board

		if self.nim_game.get_winner() is not None:
			done = True
		else:
			oppo_move = self.opponent.get_move(self.nim_game.board)
			self.nim_game.play_move(oppo_move)
			next_state = self.nim_game.board
			
			if self.nim_game.get_winner() is not None:
				done = True

		if (self.nim_game.board<0).any():
			print("SSSSS")
			exit(0)

		# print(prev_state, action, reward, next_state, done)
		return (reward, next_state, done)





class SARSANNNimAgent:

	def __init__(self, epsilon=0.01):

		get_custom_objects().update({'sine': Activation(K.sin)})
		model = Sequential()
		model.add(Dense(units=limit+1, activation='sine', input_dim=limit*n_rows))
		model.add(Dense(units=1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
		self.model = model
		self.last_state = None
		self.last_action = None
		self.x_data = None
		self.y_data = None
		self.epsilon = epsilon

	def get_action(self, last_reward, state, done):

		if np.random.binomial(1, self.epsilon):

			action = get_nims_random_move(state)

		else:

			next_states = []
			next_actions = []

			for i in range(state.shape[0]):
				for a in range(1, state[i]+1):
					next_actions.append((i, a))
					# import pdb; pdb.set_trace()
					# next_states = np.vstack((next_states, encode(state, (i, a))))
					next_states.append(encode(state, (i, a)).flatten())

			# try:
			next_states = np.array(next_states)
			next_qs = self.model.predict(next_states)
			# except Exception:
			# 	import pdb; pdb.set_trace()

			action = next_actions[next_qs.argmax()]



		if last_reward is not None:
			# import pdb; pdb.set_trace()
			next_q = self.model.predict(encode(state, action))

			if self.x_data is None:

				self.x_data = encode(self.last_state, self.last_action)
				self.y_data = np.array(last_reward + next_q).reshape((1, -1))
				self.model.fit(self.x_data, self.y_data, verbose=0)

				self.x_data = None
				self.y_data = None
				# self.y_data = np.array(last_reward).reshape((1, -1))

			else:

				self.x_data = np.vstack((self.x_data, encode(self.last_state, self.last_action)))
				self.y_data = np.vstack((self.y_data, last_reward + next_q))
				# self.y_data = np.vstack((self.y_data, last_reward))

		# print("X" if self.x_data is None else self.x_data.shape)

		if self.x_data is not None and self.x_data.shape[0] == 1000:

			# npoints_x = self.x_data.shape[0]*10

			# x_data_r = np.random.randint(0, 2, (npoints_x//2, n_rows, limit))
			# y_data_r = (x_data_r.sum(axis=1)%2).any(axis=1).reshape((-1,1)).astype(int)
			# x_data_r = x_data_r.reshape((npoints_x//2, -1))


			# x_data_0 = np.random.randint(0, 2, (npoints_x//2, n_rows-1, limit))
			# x_data_0 = np.concatenate((x_data_0, (x_data_0.sum(axis=1)%2).reshape((npoints_x//2, 1, limit))), axis=1)
			# x_data_0 = x_data_0.reshape((npoints_x//2, -1))
			# y_data_0 = np.zeros((npoints_x//2, 1))

			# x_data = np.vstack((x_data_r, x_data_0))
			# y_data = np.vstack((y_data_r, y_data_0))

			# perm = np.random.permutation(npoints_x)
			# x_data = x_data[perm]
			# y_data = y_data[perm]

			# self.model.fit(x_data, y_data, epochs=10)


			# print(self.x_data)
			# print(self.y_data)
			self.model.fit(self.x_data, self.y_data, epochs=10)
			self.x_data = None
			self.y_data = None

		self.last_action = action
		self.last_state = state.copy()

		return action


if __name__ == '__main__':
	n_total = 0
	n_wins = 0

	sarsa_agent = SARSANNNimAgent()

	while True:

		nim_env = OptimalNimEnv()
		reward, next_state, done = None, nim_env.get_state(), False

		while not done:
			action = sarsa_agent.get_action(reward, next_state, done)
			reward, next_state, done = nim_env.play_action(action)

		if nim_env.nim_game.get_winner() == 0:

			n_wins += 1

		n_total += 1

		print(n_wins, n_total, n_wins/n_total)




