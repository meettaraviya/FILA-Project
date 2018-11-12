import numpy as np
from games import *

class OptimalESSDefendingAgent:

	def get_move(self, game_state):

		n = len(game_state[0])
		coeff = 2**np.arange(n)
		partition0_value = game_state[2].dot(coeff)
		total_value = game_state[0].dot(coeff)

		if 2*partition0_value > total_value:
			return 1
		else:
			return 0

class OptimalESSAttackingAgent:

	def get_move(self, game_state):

		n = len(game_state[0])
		move = np.zeros(n, dtype=int)
		coeff = 2**np.arange(n)
		total_value = game_state[0].dot(coeff)
		partition0_value = 0

		for i in range(n):
			if partition0_value + game_state[0][i]*coeff[i] <= total_value/2:
				partition0_value += game_state[0][i]*coeff[i]
				move[i] = game_state[0][i]
			else:
				move[i] = min(round((total_value/2 - partition0_value)/coeff[i]), game_state[0][i])
				break

		return move


if __name__ == "__main__":
	init_board = np.array([8,3,0,0,0], dtype=int)
	print(init_board)
	game = ESSGame(init_board)
	agents = [OptimalESSAttackingAgent(), OptimalESSDefendingAgent()]

	while game.get_winner() is None:

		# print(game.game_state)
		move = agents[game.game_state[1]].get_move(game.game_state)
		game.play_move(move)
		# print(move)

	# print(game.game_state)
	print(["Attacker wins!", "Defender winds!"][game.get_winner()])
