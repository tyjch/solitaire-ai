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
	'num_workers' : 0,

	'env'         : 'solitaire_env',
	'model'       : {
		'custom_model': 'solitaire_model',
	},

	#'vf_loss_coeff'           : 0.5,
	'evaluation_num_episodes' : 20,
	'evaluation_config'       : {
		'explore' : False
	},
	'explore'                 : True,
	'exploration_config'      : {
		'type' : 'SoftQ' ,
		'temperature' : 0.5,
	},

	"input"              : {
		"/Users/tylerchurchill/Desktop/solitaire-ai/solver/sample_batches/*.json" : 0.50,
		"sampler" : 0.50,
	},
    "input_evaluation"   : ["is", "wis"],
    "beta"               : 1.0,
    "vf_coeff"           : 1.0,
    "postprocess_inputs" : False,
    "batch_mode"         : "complete_episodes",
    "lr"                 : 1e-4,
    "train_batch_size"   : 200,
    "replay_buffer_size" : 200,
    "learning_starts"    : 0,
}

stop = {
	'timesteps_total' : 1000000,
}

results = tune.run(
		"MARWIL",
		num_samples = 1,
		stop        = stop,
		config      = config,
		verbose     = 1,
		local_dir   = "/Users/tylerchurchill/Desktop/solitaire-ai/experiments/MARWIL",
		name        = 'solitaire',
)

ray.shutdown()