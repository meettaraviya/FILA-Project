import numpy as np

class ESSGame:

	def __init__(self, init_board):

		self.game_state = [init_board, 0, None] # 0 = Attacker's turn

	def play_move(self, move):

		if self.game_state[1] == 0: # Attacker specifies no. of tokens from each row to keep in one partition
			self.game_state[2] = move
		else:
			if move == 0: # Defender keeps the partition specified by attacker
				self.game_state[0] = self.game_state[2]
			else:
				self.game_state[0] = self.game_state[0] - self.game_state[2]

			self.game_state[0] = np.roll(self.game_state[0], 1)
			self.game_state[2] = None

		self.game_state[1] = 1 - self.game_state[1]

	def get_winner(self):

		if self.game_state[0][-1] > 0:
			return 0 # Attacker
		elif self.game_state[0].sum() == 0:
			return 1 # Defender
		else:
			return None

class NimsGame:

	def __init__(self, init_state):

		self.board = init_state
		self.player = 0

	def play_move(self, move):

		self.board[move[0]] -= move[1]
		self.player = 1 - self.player

	def get_board(self):
		return self.board

	def get_player(self):
		return self.player

	def get_winner(self):
		if self.board.sum() == 0:
			return 1 - self.player
		else:
			return None

	def print_game(self):
		print self.board,self.player