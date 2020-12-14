import pyspiel, gym
import numpy as np
from gym import spaces
from collections import OrderedDict
from itertools import chain
from colors import red, blue, yellow
from pprint import pprint

# region Observation Class

cards = [
	'Hidden',
	'As','2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'Ts', 'Js', 'Qs', 'Ks',
	'Ah','2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'Th', 'Jh', 'Qh', 'Kh',
	'Ac','2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'Tc', 'Jc', 'Qc', 'Kc',
	'Ad','2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'Td', 'Jd', 'Qd', 'Kd',
]
suits = ('s', 'h', 'c', 'd')
ranks = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')

class Observation:

	def __init__(self, raw_observation):
		self.raw_observation = raw_observation
		observation = list(self.raw_observation['observation'])

		self.foundation_tensors = [
			observation[0:14],
			observation[14:28],
			observation[28:42],
			observation[42:56],
		]
		self.tableau_tensors    = [
			observation[56:115],
			observation[115:174],
			observation[174:233],
			observation[233:292],
			observation[292:351],
			observation[351:410],
			observation[410:469],
		]
		self.waste_tensors      = [
			observation[469:522],
			observation[522:575],
			observation[575:628],
			observation[628:681],
			observation[681:734],
			observation[734:787],
			observation[787:840],
			observation[840:893],
			observation[893:946],
			observation[946:999],
			observation[999:1052],
			observation[1052:1105],
			observation[1105:1158],
			observation[1158:1211],
			observation[1211:1264],
			observation[1264:1317],
			observation[1317:1370],
			observation[1370:1423],
			observation[1423:1476],
			observation[1476:1529],
			observation[1529:1582],
			observation[1582:1635],
			observation[1635:1688],
			observation[1688:1741],
		]

		self.foundations = self.tensor_to_foundation()
		self.tableaus    = self.tensor_to_tableau()
		self.waste       = self.tensor_to_waste()

	def tensor_to_foundation(self):
		foundations = []
		for tensor, suit in zip(self.foundation_tensors, suits):
			try:
				foundation = [ranks[i] + suit for i in range(tensor.index(1))]
				foundations.append(foundation)
			except ValueError:
				print('ERROR:')
				pprint(self.raw_observation)
				pprint(self.foundation_tensors)
				print('tensor:', tensor)
				print('suit:', suit)
				raise ValueError
		return foundations

	def tensor_to_tableau(self):
		tableaus = []
		for index, tensor in enumerate(self.tableau_tensors):
			hidden_cards = ['🂠' for i in tensor[0:6] if i == 1]
			shown_cards  = [cards[i] for i, v in enumerate(tensor[6:]) if v == 1]
			shown_cards.sort(key=lambda n : list(reversed(ranks)).index(n[0]))
			tableaus.append(hidden_cards + shown_cards)
		return tableaus

	def tensor_to_waste(self):
		waste = []
		for tensor in self.waste_tensors:
			try:
				waste_card = cards[tensor.index(1)]
				waste.append(waste_card)
			except ValueError:
				continue
		return waste

	def foundations_to_tensor(self):
		return list(self.raw_observation['observation'])[0:56]

	def tableaus_to_tensor(self):
		return list(sorted(self.tableau_tensors))

	def waste_to_tensor(self):
		return self.waste_tensors

	def get_processed_obs(self):
		foundations = list(chain(self.foundations_to_tensor()))
		tableaus = list(chain.from_iterable(self.tableaus_to_tensor()))
		waste = list(chain.from_iterable(self.waste_to_tensor()))

		output = OrderedDict({
			'action_mask' : self.raw_observation['action_mask'],
			'observation' : np.array(foundations + tableaus + waste),
		})

		return output

# endregion


class SolitaireEnv(gym.Env):

	def __init__(self, env_config):
		parameters = {
			'is_colored'  : pyspiel.GameParameter(env_config.get('is_colored', True)),
			'depth_limit' : pyspiel.GameParameter(env_config.get('depth_limit', 300))
		}
		self.game  = pyspiel.load_game('solitaire', parameters)
		self.state = None

		self.action_space = spaces.Discrete(self.game.num_distinct_actions())
		self.observation_space = spaces.Dict({
			'action_mask': spaces.MultiBinary(self.game.num_distinct_actions()),
			'observation': spaces.MultiBinary(self.game.observation_tensor_size())
		})

		self.deterministic_chance = env_config.get('deterministic_chance', False)
		self.should_terminate = False

	def execute_chance_step(self, action):
		if self.state.is_chance_node:
			if self.deterministic_chance:
				pprint(dir(self.state))
				self.state.apply_action(action)
			else:
				raise Warning('`deterministic_chance` is False')
		else:
			raise Warning('`Cannot call `execute_chance_step` on non-chance node')

	def _execute_chance_steps(self):
		r = 0
		while self.state.is_chance_node() and not self.state.is_terminal():
			try:
				chose_action = np.random.choice(self.state.legal_actions())
				self.state.apply_action(chose_action)
			except ValueError:
				print(red('ValueError: No chance actions available'))
				self.should_terminate = True
			r += self.state.rewards()[0]
		return r

	def _get_actions_mask(self):
		empty_array = np.zeros(shape=self.game.num_distinct_actions(), dtype=np.int8)
		if self.state.is_player_node():
			for action in self.state.legal_actions(0):
				empty_array[action] = 1
		return empty_array

	def render(self, mode='human'):
		if mode == 'human':
			if self.state.is_chance_node():
				print(yellow('--' * 50))
				print(yellow('CHANCE NODE'))
			elif self.state.is_player_node():
				print(blue('--' * 50))
				print(blue('PLAYER NODE'))
			else:
				print(red('--' * 50))
				print(red('TERMINAL NODE'))

			print(self.state)
			print('HISTORY :')
			for action in reversed(self.state.history()[-5:]):
				print(' ', self.state.action_to_string(action))

	def reset(self):
		self.should_terminate = False
		self.state = self.game.new_initial_state()
		if not self.deterministic_chance:
			self._execute_chance_steps()
		return self.observation

	def step(self, action):
		self.state.apply_action(action)
		if not self.deterministic_chance:
			self._execute_chance_steps()
		return self.observation, self.reward, self.done, self.info

	@property
	def observation(self):
		try:
			observation = OrderedDict({
				'action_mask': self._get_actions_mask(),
				'observation': np.array(self.state.observation_tensor(), dtype=np.int8)
			})
		except pyspiel.SpielError:
			observation = OrderedDict({
				'action_mask': self._get_actions_mask(),
				'observation': np.zeros(shape=self.game.observation_tensor_size(), dtype=np.int8)
			})

		assert self.observation_space.contains(observation)
		return observation

	@property
	def reward(self):
		return self.state.player_reward(0)

	@property
	def done(self):
		return self.state.is_terminal() or self.should_terminate

	@property
	def info(self):
		return {}


if __name__ == '__main__':
	np.random.seed(2)
	env = SolitaireEnv({})
	obs, reward, done, info = env.reset(), 0, False, {}

	while not done:
		print(env.state)
		obs_view = Observation(obs)
		obs_view.get_processed_obs()

		legal_actions = [i for (i, a) in enumerate(obs['action_mask']) if a != 0]
		chosen_action = np.random.choice(legal_actions)
		obs, reward, done, info = env.step(chosen_action)

