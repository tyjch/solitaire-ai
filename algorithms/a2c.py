import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from env import SolitaireEnv
from model import SolitaireModel

ray.init(local_mode=True)
register_env('solitaire_env', lambda _: SolitaireEnv({}))
ModelCatalog.register_custom_model('solitaire_model', SolitaireModel)

config = {
	'framework'   : 'torch',
	'log_level'   : 'INFO',
	'num_workers' : 1,

	'env'         : 'solitaire_env',
	'model'       : {
		'custom_model': 'solitaire_model',
	},

	'vf_loss_coeff' : 0.5,
	'evaluation_num_episodes' : 20,
	'evaluation_config' : {
		'explore' : False
	},
	'explore' : True,
	'exploration_config' : {
		'type' : 'SoftQ' ,
		'temperature' : 0.5,
	},

	'batch_mode' : 'complete_episodes',
	'output'     : '/Users/tylerchurchill/Desktop/solitaire-ai/solver/sample_batches',
}

stop = {
	'timesteps_total' : 100000,
}

results = tune.run(
		"A2C",
		num_samples = 1,
		stop        = stop,
		config      = config,
		verbose     = 1,
		local_dir   = "/Users/tylerchurchill/Desktop/solitaire-ai/experiments/A2C",
		name        = 'solitaire',
)

ray.shutdown()