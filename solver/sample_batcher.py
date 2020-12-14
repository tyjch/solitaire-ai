import glob
import numpy as np
from colors import blue
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from env import SolitaireEnv
from solver import TRANSLATOR, process_experience

INPUT_PATH  = tuple('/Users/tylerchurchill/Desktop/solitaire-ai/solver/Solvitaire/output.txt')
OUTPUT_PATH = '/Users/tylerchurchill/Desktop/solitaire-ai/solver/sample_batches'

def generate_sample_batches(experience_paths=INPUT_PATH, output_path=OUTPUT_PATH) -> None:

	batch_builder = SampleBatchBuilder()
	writer        = JsonWriter(output_path)

	for eps_id, experience_path in enumerate(experience_paths):

		print(blue(f'[info] Generating sample batch from {experience_path}'))

		moves = process_experience(input_path=experience_path)
		env   = SolitaireEnv({
			'depth_limit'         : len(moves),
			'deterministic_chance': True,
		})
		prep  = get_preprocessor(env.observation_space)(env.observation_space)
		print(f'The preprocessor is {prep}')

		obs         = env.reset()
		prev_action = np.zeros_like(env.action_space.sample())
		prev_reward = 0
		t           = 0

		for m in moves:
			action = TRANSLATOR[m]
			print(env.state)

			if 1 <= action <= 52:
				env.execute_chance_step(action)
			else:
				new_obs, rew, done, info = env.step(action)
				batch_builder.add_values(
					t            = t,
					eps_id       = eps_id,
					agent_index  = 0,
					obs          = prep.transform(obs),
					actions      = action,
					action_prob  = 1.0,
					action_logp  = -1.0,
					rewards      = rew,
					prev_actions = prev_action,
					prev_rewards = prev_reward,
					dones        = done,
					infos        = {'path':experience_path, **info},
					new_obs      = prep.transform(new_obs)
				)

				obs         = new_obs
				prev_action = action
				prev_reward = rew
				t          += 1

		writer.write(batch_builder.build_and_reset())


if __name__ == "__main__":
	paths = glob.glob('./Solvitaire/experiences/*.txt')
	generate_sample_batches(experience_paths=paths)

