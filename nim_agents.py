from keras.models import Sequential
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, Dense, Maximum
from keras import optimizers
import keras
import  itertools

import operator
import numpy as np
from games import *

n_rows = 4
limit = 4

def NimsGameRandomMove(board):

	move = [None, None]
	move[0] = np.random.choice(np.where(board>0)[0])
	move[1] = np.random.randint(1, 1+board[move[0]])
	return move


class EpsilonRandomNimAgent:
	def __init__(self, epsilon):
		self.epsilon = epsilon
	def get_move(self, board):

		move = [None, None]
		if np.random.binomial(1, self.epsilon) == 1:
			return NimsGameRandomMove(board)

		else:
			xor = np.bitwise_xor.reduce(board)
			if xor != 0:
				move = [None, None]
				move[0] = np.where(np.bitwise_xor(board, xor)<board)[0][0]
				move[1] = board[move[0]] - (board[move[0]]^xor)
				return move

			else:
				return NimsGameRandomMove(board)
	
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
			return NimsGameRandomMove(board)
	
	def gameOver(self,win):
		pass

def NN(n_rows, limit):
	get_custom_objects().update({'sine': Activation(K.sin)})
	model = Sequential()
	model.add(Dense(units=(limit+1)*n_rows, activation='sine', input_dim=n_rows))
	model.add(Dense(units=limit+1, activation='sine'))
	model.add(Dense(units=1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.001), metrics=['accuracy'])
	return model

def NN_ohe(n_rows, limit):
	get_custom_objects().update({'sine': Activation(K.sin)})
	model = Sequential()
	model.add(Dense(units=limit+1, activation='sine', input_dim=limit*n_rows))
	model.add(Dense(units=1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.001), metrics=['accuracy'])
	return model

def getNimActions(board):
	actions = []
	for i in range(len(board)):
		for j in range(1, board[i]+1):
			actions.append((i,j))
	return actions


def getNextState(board,actions):
	nextstates = []
	for act in actions:		
		b = board.copy()
		b[act[0]] -= act[1] 
		nextstates.append(b.copy())
	return nextstates


class BatchMCNNNimAgent:

	def __init__(self, batch_size=100, n_rows=10, limit=5):
		self.steps_since_update = 0
		self.batch_size = batch_size
		self.n_rows	= n_rows
		self.limit = limit
		self.model = NN_ohe(self.n_rows, self.limit)
		self.time_since_train = 0
		self.history = [[]]
		self.completed_runs = 0

	def NN_trans(self,states):
		stts = [ list(itertools.chain.from_iterable([list(bin(i)[2:].zfill(self.limit)) for i in z])) for z in states]
		# for x in stts:
		# 	x.append(1)
		# import pdb; pdb.set_trace()
		return np.array(stts,dtype= int)


	def train(self):
		x_train = [h[:-1] for h in self.history[:self.completed_runs]]
		y_train = [ [h[-1]]*(len(h)-1) for h in self.history[:self.completed_runs]]
		x_train = np.array(list(itertools.chain.from_iterable(x_train)),dtype= int)
		y_train = np.array(list(itertools.chain.from_iterable(y_train)),dtype= int)

		# import pdb; pdb.set_trace()
		# x = np.array([ np.array(list(itertools.chain.from_iterable( [ list(bin(i)[2:].zfill(limit)) for i in _x ])) ,dtype=int) for _x in x_train])
		x_train = self.NN_trans(x_train)

			
		self.model.fit(x_train,y_train, epochs=10, verbose=2)

	def getBestNextStateIdx(self,nextstates):

		test = self.NN_trans(nextstates)
		scores = self.model.predict(test)
		# print("BSK",scores)
		# print(max(enumerate(scores), key=operator.itemgetter(1)))
		index, value = max(enumerate(scores), key=operator.itemgetter(1))
		return index
	
	def get_move(self, board):
		
		actions = getNimActions(board)
		nextStates = getNextState(board,actions)
		# print(board,actions,nextStates)
		self.time_since_train += 1
		# import pdb; pdb.set_trace()
		# print(self.history)
		self.history[len(self.history)-1].append(board.copy())
		# import pdb; pdb.set_trace()
		# print(self.history)

		if self.time_since_train == 100 and self.completed_runs>0:
			self.time_since_train = 0
			self.train()
			# self.history = [[]]
			# self.completed_runs = 0

		return actions[self.getBestNextStateIdx(nextStates)]

	def gameOver(self,win):
		self.history[-1].append(win)
		self.completed_runs = self.completed_runs + 1
		self.history.append([])

if __name__ == "__main__":
	# print(init_board)
	np.random.seed(0)
	agents = [ EpsilonRandomNimAgent(1),BatchMCNNNimAgent(n_rows=n_rows)]
	# agents = [OptimalNimAgent(),OptimalNimAgent()]
	wins = 0
	games = [NimsGame(np.random.randint(2**limit, size=n_rows).copy()) for i in range(1000)]

	# winner = [np.bitwise_xor.reduce(games[i].board) == 0 for i in range(1000)]
	# print(sum(winner))
	
	for _ in range(1000000):
		if (_)%1000 == 999:
			# agents[0] = EpsilonRandomNimAgent(1)
			print("Wins=", wins)
			wins = 0
			agents[0].epsilon = min(1000/(1+_), 1)
		game = NimsGame(np.random.randint(2**limit, size=n_rows))

		while game.get_winner() is None:
			# game.print_game()
			move = agents[game.get_player()].get_move(game.get_board())
			game.play_move(move)

		agents[0].gameOver(1-game.get_winner())
		agents[1].gameOver(game.get_winner())
		wins += game.get_winner()

	# print("Total Wins=", sum(winner))
