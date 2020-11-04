import pyspiel, gym
import numpy as np
from gym import spaces
from collections import OrderedDict
from colors import red, blue, yellow


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

		self.should_terminate = False

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
		self._execute_chance_steps()
		return self.observation

	def step(self, action):
		self.state.apply_action(action)
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
		return self.state.player_reward(0) - 1.0

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
		legal_actions = [i for (i, a) in enumerate(obs['action_mask']) if a != 0]
		chosen_action = np.random.choice(legal_actions)
		obs, reward, done, info = env.step(chosen_action)
