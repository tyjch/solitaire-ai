import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from env import SolitaireEnv
from model import SolitaireModel

# TODO: Doesn't work because `ValueError: Expected output shape of [None, 256], got torch.Size([1, 205])`

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
}

stop = {
	'timesteps_total' : 10000,
}

results = tune.run(
		"DQN",
		num_samples = 1,
		stop        = stop,
		config      = config,
		verbose     = 1,
		local_dir   = "/Users/tylerchurchill/Desktop/solitaire-ai/experiments/DQN",
		name        = 'solitaire',
)

ray.shutdown()