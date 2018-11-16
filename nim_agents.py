import numpy as np
from games import *

def NimsGameRandomMove(game_state):

	move = [None, None]
	move[0] = np.random.choice(np.where(game_state[0]>0)[0])
	move[1] = np.random.randint(1, 1+game_state[0][move[0]])
	return move

class OptimalNimAgent:

	def get_move(self, game_state):
		xor = np.bitwise_xor.reduce(game_state[0])
		if xor != 0:
			move = [None, None]
			move[0] = np.where(np.bitwise_xor(game_state[0], xor)<game_state[0])[0][0]
			move[1] = game_state[0][move[0]] - (game_state[0][move[0]]^xor)
			return move

		else:
			return NimsGameRandomMove(game_state)

class BatchSarsaNonLinearApproximationNimAgent:

	def __init__(self, batch_size=100, n_rows=10):

		self.steps_since_update = 0

		from keras.models import Sequential
		from keras import backend as K
		from keras.utils.generic_utils import get_custom_objects
		
		get_custom_objects().update({'sine': Activation(K.sin)})

		self.model = Sequential()
		self.model.add(Dense(units=10, activation='sigmoid', input_dim=n_rows))
		self.model.add(Dense(units=10, activation='sine'))
		self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

	def get_move(self, game_state):





if __name__ == "__main__":
	init_board = np.array([8,3,2,4,1,12], dtype=int)
	print(init_board)
	game = NimsGame(init_board)
	agents = [OptimalNimAgent(), OptimalNimAgent()]

	while game.get_winner() is None:

		print(game.game_state)
		move = agents[game.game_state[1]].get_move(game.game_state)
		game.play_move(move)
		# print(move)

	# print(game.game_state)
	print(["Player 1 wins!", "Player 2 wins!"][game.get_winner()])
