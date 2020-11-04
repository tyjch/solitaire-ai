import numpy as np
from gym.spaces import Box
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.numpy import LARGE_INTEGER

torch, nn = try_import_torch()

class SolitaireModel(DQNTorchModel):

    def __init__(
		    self,
            obs_space,
		    action_space,
		    num_outputs,
		    model_config,
		    name,
            true_obs_shape=(1741,1),
		    action_embed_size=205,
		    **kwargs
    ):

        true_obs_shape    = model_config['custom_model_config'].get('true_obs_shape', true_obs_shape)
        action_embed_size = model_config['custom_model_config'].get('action_embed_size', action_embed_size)

        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        self.action_embed_model = TorchFC(
            obs_space    = Box(-1.0, 1.0, shape=true_obs_shape),
            action_space = action_space,
            num_outputs  = action_embed_size,
            model_config = model_config,
            name         = name + '_action_embed'
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask  = input_dict['obs']['action_mask']
        action_embed, _ = self.action_embed_model({
            'obs' : input_dict['obs']['observation']
        })

        inf_mask = torch.clamp(
            torch.log(action_mask),
            -float(LARGE_INTEGER),
            float("inf")
        )

        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

    def import_from_h5(self, h5_file):
        pass

'''
tf1, tf, tfv = try_import_tf()

class SolitaireModelTF(TFModelV2):

    def __init__(
		    self,
            obs_space,
		    action_space,
		    num_outputs,
		    model_config,
		    name,
            true_obs_shape=(1741,1),
		    action_embed_size=205,
		    **kwargs
    ):

        super(SolitaireModelTF, self).__init__(
			obs_space,
	        action_space,
	        num_outputs,
	        model_config,
	        name,
	        **kwargs
		)

        self.action_embed_model = FullyConnectedNetwork(
	        Box(np.NINF, np.PINF, shape=true_obs_shape),
	        action_space,
	        action_embed_size,
	        model_config,
	        name + "_action_embed"
        )

        self.obs_space = obs_space
        self.register_variables(self.action_embed_model.variables())


    def forward(self, input_dict, state, seq_lens):
        # Compute the predicted action embedding
	    action_embed, _ = self.action_embed_model({
		    "obs": input_dict["obs"]["cart"]
	    })

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
	    # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
	    intent_vector = tf.expand_dims(action_embed, 1)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
	    action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)

        # Mask out invalid actions (use tf.float32.min for stability)
	    inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

    def import_from_h5(self, h5_file):
        pass
'''