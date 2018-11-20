from keras.models import Sequential
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, Dense, Maximum
import keras

import operator
import numpy as np
from games import *

def NimsGameRandomMove(board):

	move = [None, None]
	move[0] = np.random.choice(np.where(board>0)[0])
	move[1] = np.random.randint(1, 1+board[move[0]])
	return move

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

def NN(n_rows):
	get_custom_objects().update({'sine': Activation(K.sin)})

	model = Sequential()
	model.add(Dense(units=n_rows, activation='sigmoid', input_dim=n_rows))
	model.add(Dense(units=n_rows, activation='sine'))
	# model.compile(loss=keras.losses.categorical_crossentropy,
	# sgd =keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def getNimActions(board):
	actions = []
	for i in range(len(board)):
		for j in range(board[i]+1):
			actions.append((i,j))
	return actions


def getNextState(board,actions):
	nextstates = []
	b = board.copy()
	for act in actions:		
		b[act[0]] -= act[1] 
		nextstates.append(b)
	return nextstates

class BatchMCNNNimAgent:

	def __init__(self, batch_size=100, n_rows=10):
		self.steps_since_update = 0
		self.batch_size = batch_size
		self.n_rows	= 10
		self.model = NN(self.n_rows)
		self.time_since_train = 0
		self.history = [[]]
		self.completed_runs = 0

	def train(self):
		x_train = [h[:-1] for h in self.history[:completed_runs]]
		y_train = [ [h[-1]]*(len(h)-1) for h in self.history[:completed_runs]]
		x_train = np.array(list(itertools.chain.from_iterable(x_train)),dtype= int)
		y_train = np.array(list(itertools.chain.from_iterable(y_train)),dtype= int)

		self.model.fit(x_train,y_train,epochs=10)

	def getBestNextStateIdx(self,nextstates):
		if self.time_since_train == 100 and self.completed_runs>0:
			self.time_since_train = 0
			self.train()
			self.history = [[]]
			self.completed_runs = 0

		test = np.array(nextstates, dtype= int)
		scores = self.model.predict(test)
		index, value = max(enumerate(scores), key=operator.itemgetter(1))
		return index
	
	def get_move(self, board):
		actions = getNimActions(board)
		nextStates = getNextState(board,actions)
		self.time_since_train = self.time_since_train +1
		self.history[-1].append(board)
		return actions[self.getBestNextStateIdx(nextStates)]

	def gameOver(self,win):
		self.history[-1].append(win)
		self.completed_runs = self.completed_runs + 1
		self.history.append([])

if __name__ == "__main__":
	init_board = np.array([8,3,2,4,1,12], dtype=int)
	print(init_board)
	game = NimsGame(init_board)
	agents = [OptimalNimAgent(), BatchMCNNNimAgent()]

	while game.get_winner() is None:
		game.print_game()
		move = agents[game.get_player()].get_move(game.get_board())
		game.play_move(move)
		# print(move)

	agents[0].gameOver(1-game.get_winner())
	agents[1].gameOver(game.get_winner())

	# print(game.game_state)
	print(["Player 1 wins!", "Player 2 wins!"][game.get_winner()])
