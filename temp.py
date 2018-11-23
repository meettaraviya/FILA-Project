
class BatchMCNNNimAgent:

	def __init__(self, batch_size=100, n_rows=10, limit=5):
		# self.steps_since_update = 0
		self.batch_size = batch_size
		self.n_rows	= n_rows
		self.limit = limit
		self.n_steps = 50
		self.total_moves = 0
		# self.model = NN_ohe(self.n_rows, self.limit)
		self.time_since_train = 0
		self.history = [[]]
		self.completed_runs = 0
		self.actcrtc = ActorCritic()

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
	
	def get_id_from_tuple(self,move):
		return move[0]*(2**self.limit) + move[1]

	def get_move(self, board):
		s = torch.from_numpy(board).float().unsqueeze(0)
		action_probs = self.actcrtc.get_action_probs(Variable(s))
        action = action_probs.multinomial().data[0][0]
		
		self.time_since_train += 1
        self.history[len(self.history)-1].append(board.copy())
		
		if self.time_since_train == self.n_steps:
			self.time_since_train = 0
			self.history = [[]]
			
			if len(self.history)>1:	##TBD
				self.train()

			# self.completed_runs = 0

		
		return self.get_id_from_tuple(action)
        

	def gameOver(self,win):
		self.history[-1].append(win)
		self.completed_runs = self.completed_runs + 1
		self.history.append([])