from collections import OrderedDict
from typing import Optional, List, Dict, cast, Tuple
import numpy as np

import gym
import math
import torch
import torch.nn as nn
from gym.spaces import Dict as SpaceDict

from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)
from transformers import T5EncoderModel, AutoTokenizer
from utils.string_utils import convert_byte_to_string
from utils.nn_utils import debug_model_info
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.algorithms.onpolicy_sync.policy import ObservationType, DistributionType
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.utils.model_utils import FeatureEmbedding
from allenact.utils.system import get_logger
from training.online.third_party_models.llama.model import (
    TransformerDecoder as LLAMATransformerDecoder,
)
from training.online.third_party_models.llama.model import ModelArgs as LLAMAModelArgs
from utils.bbox_utils import get_best_of_two_bboxes


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, position):
        """
        Args:
            position: Tensor, shape [batch_size, seq_len]
        """
        B, L = position.shape
        position = position.unsqueeze(-1)  # BxLx1
        pe = torch.zeros([B, L, self.d_model], device=position.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe


class DinoLLAMATxNavActorCritic(VisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_tx_layers=3,
        num_tx_heads=8,
        text_embed_size=512,
        add_prev_actions=False,
        add_prev_action_null_token=False,
        action_embed_size=512,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        rgb_dino_preprocessor_uuid: Optional[str] = None,
        goal_dims: int = 512,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (384, 512),
        combiner_hidden_out_dims: int = 512,
        combiner_nhead: int = 8,
        combiner_layers: int = 3,
        max_steps: int = 1000,
        max_steps_for_training: int = 128,
        time_step_uuid: Optional[str] = None,
        initial_kv_cache_shape: Tuple[int, int] = (128, 32),
        traj_idx_uuid: Optional[str] = None,
        traj_max_idx: Optional[int] = None,
        relevant_object_box_uuid: Optional[str] = None,
        accurate_object_box_uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            **kwargs,
        )

        assert action_embed_size == combiner_hidden_out_dims
        self.time_step_counter = 0
        self.traj_idx_uuid = traj_idx_uuid
        self.traj_max_idx = traj_max_idx
        self.relevant_object_box_uuid = relevant_object_box_uuid
        self.accurate_object_box_uuid = accurate_object_box_uuid

        self.text_embed_size = text_embed_size
        self.max_steps = max_steps
        self.max_steps_for_training = max_steps_for_training
        self.time_step_uuid = time_step_uuid
        self.goal_sensor_uuid = goal_sensor_uuid
        if rgb_dino_preprocessor_uuid is not None:
            dino_preprocessor_uuid = rgb_dino_preprocessor_uuid
            self.goal_visual_encoder = DinoTxGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                dino_preprocessor_uuid,
                relevant_object_box_uuid,
                accurate_object_box_uuid,
                goal_dims,
                dino_compressor_hidden_out_dims,
                combiner_hidden_out_dims=combiner_hidden_out_dims,
                combiner_heads=combiner_nhead,
                combiner_layers=combiner_layers,
            )

        self.state_encoders_time: Optional[nn.ModuleDict] = None
        self.state_encoders_linear: Optional[nn.ModuleDict] = None
        self.state_encoders_text: Optional[nn.ModuleDict] = None
        self.create_tx_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            text_embed_size=text_embed_size,
            num_tx_layers=num_tx_layers,
            num_tx_heads=num_tx_heads,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
            initial_kv_cache_shape=initial_kv_cache_shape,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()
        debug_model_info(self, use_logger=False)

    def sampler_select(self, keep: list):
        for key, model in self.state_encoders.items():
            if hasattr(model, "sampler_select"):
                model.sampler_select(keep)

    def create_tx_state_encoders(
        self,
        obs_embed_size: int,
        text_embed_size: int,
        prev_action_embed_size: int,
        num_tx_layers: int,
        num_tx_heads: int,
        add_prev_actions: bool,
        add_prev_action_null_token: bool,
        initial_kv_cache_shape: Tuple[int, int],
    ):
        tx_input_size = obs_embed_size
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )

        state_encoders_params = LLAMAModelArgs(
            dim=obs_embed_size,
            n_layers=num_tx_layers,
            n_heads=num_tx_heads,
            vocab_size=obs_embed_size,
            max_batch_size=initial_kv_cache_shape[1],
            max_seq_len=initial_kv_cache_shape[0],
        )

        state_encoders_linear = OrderedDict()
        state_encoders_time = OrderedDict()
        state_encoders_text = OrderedDict()
        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            for aux_uuid in self.auxiliary_uuids:
                state_encoders_linear[aux_uuid] = nn.Linear(tx_input_size, self._hidden_size)
                state_encoders_time[aux_uuid] = PositionalEncoder(
                    self._hidden_size, max_len=self.max_steps
                )
                if self.goal_sensor_uuid is not None:
                    state_encoders_text[aux_uuid] = nn.Sequential(
                        nn.Linear(text_embed_size, self._hidden_size),
                        nn.LayerNorm(self._hidden_size),
                        nn.ReLU(),
                    )
                state_encoders[aux_uuid] = LLAMATransformerDecoder(state_encoders_params)
            # create fusion model
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )

        else:  # single belief model
            state_encoders_linear["single_belief"] = nn.Linear(tx_input_size, self._hidden_size)
            state_encoders_time["single_belief"] = PositionalEncoder(
                self._hidden_size, max_len=self.max_steps
            )
            if self.goal_sensor_uuid is not None:
                state_encoders_text["single_belief"] = nn.Sequential(
                    nn.Linear(text_embed_size, self._hidden_size),
                    nn.LayerNorm(self._hidden_size),
                    nn.ReLU(),
                )
            state_encoders["single_belief"] = LLAMATransformerDecoder(state_encoders_params)

        self.state_encoders_linear = nn.ModuleDict(state_encoders_linear)
        self.state_encoders_time = nn.ModuleDict(state_encoders_time)
        if self.goal_sensor_uuid is not None:
            self.state_encoders_text = nn.ModuleDict(state_encoders_text)
        self.state_encoders = nn.ModuleDict(state_encoders)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(len(self.belief_names), self.belief_names)
        )

    def _recurrent_memory_specification(self):
        return None

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)

    def compute_total_grad_norm(self):
        with torch.no_grad():
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm**2
            total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        ###
        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        ###
        obs_embeds, text_feats = self.forward_encoder(observations)

        ###
        # 1.2 use embedding model to get prev_action embeddings
        ###
        if self.prev_action_embedder.input_size == self.action_space.n + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)

        joint_embeds = obs_embeds + prev_actions_embeds

        ###
        # 2. use Transformers to get single/multiple beliefs
        ###
        beliefs_input_dict = {}
        # 2.1 prepare input for each belief model via linear projection
        for key, model in self.state_encoders_linear.items():
            beliefs_input_dict[key] = model(joint_embeds)
        # 2.2 add positional encoding
        for key, model in self.state_encoders_time.items():
            beliefs_input_dict[key] = (
                model(observations[self.time_step_uuid]) + beliefs_input_dict[key]
            )
        # 2.3 causal transformer decoder to get beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            # reset time step counter 1) if there are multiple time steps in the input,
            # indicating it is at the update stage; and 2) if the max_steps is reached
            if joint_embeds.shape[0] > 1 or self.time_step_counter >= self.max_steps:
                self.time_step_counter = 0
            x = beliefs_input_dict[key].permute(1, 0, 2)
            if self.traj_idx_uuid is None:
                # no dynamic attention causal mask as the trajectory index is not available
                mask = None
            elif joint_embeds.shape[0] == 1:
                # construct dynamic attention causal mask for single time step input (rollout stage)
                timesteps = observations[self.time_step_uuid].permute(1, 0)  # bs, nsteps
                epi_start = torch.clamp(self.time_step_counter - timesteps, min=0).expand(
                    -1, self.time_step_counter + 1
                )  # bs, 1
                step_range = torch.arange(0, self.time_step_counter + 1).to(device=epi_start.device)

                mask = (epi_start <= step_range).unsqueeze(1).unsqueeze(1)
            else:
                # construct dynamic attention causal mask for multiple time steps input (update stage)
                traj_idx: torch.Tensor = observations[self.traj_idx_uuid].permute(1, 0)
                mask = traj_idx[:, :, None] == traj_idx[:, None, :]
                mask = torch.tril(mask)
                mask = mask.unsqueeze(1)  # type: ignore
            # forward causal transformer decoder
            y = model(x, self.time_step_counter, mask)
            beliefs_dict[key] = y.permute(1, 0, 2)
            if joint_embeds.shape[0] == 1:
                self.time_step_counter += 1

        ###
        # 3. fuse beliefs for multiple belief models
        ###
        beliefs, task_weights = self.fuse_beliefs(beliefs_dict, obs_embeds)  # fused beliefs

        ###
        # 4. prepare output
        ###
        extras = (
            {
                aux_uuid: {
                    "beliefs": (beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid] if aux_uuid in self.aux_models else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        total_norm = self.compute_total_grad_norm()
        extras["total_norm"] = torch.Tensor([total_norm])

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )

        return actor_critic_output, memory


class DinoTxGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        dino_preprocessor_uuid: str,
        relevant_object_box_uuid: str = None,
        accurate_object_box_uuid: str = None,
        goal_embed_dims: int = 512,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (384, 512),
        combiner_hidden_out_dims: int = 512,
        combiner_layers: int = 3,
        combiner_heads: int = 8,
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.dino_uuid = dino_preprocessor_uuid
        self.relevant_object_box_uuid = relevant_object_box_uuid
        self.accurate_object_box_uuid = accurate_object_box_uuid
        self.goal_embed_dims = goal_embed_dims
        self.dino_hid_out_dims = dino_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        if goal_sensor_uuid is not None:
            self.goal_space = observation_spaces.spaces[self.goal_uuid]
            self.text_goal_encoder = T5EncoderModel.from_pretrained("google/flan-t5-small")
            self.text_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.text_adapter = nn.Sequential(
                nn.Linear(512, self.goal_embed_dims), nn.LayerNorm(self.goal_embed_dims), nn.ReLU()
            )

        self.fusion_token = nn.Parameter(0.1 * torch.rand(self.goal_embed_dims))

        self.blind = self.dino_uuid not in observation_spaces.spaces
        if not self.blind:
            self.dino_tensor_shape = observation_spaces.spaces[self.dino_uuid].shape
            self.dino_compressor = nn.Sequential(
                nn.Conv2d(self.dino_tensor_shape[-1], self.dino_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.dino_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.target_obs_combiner = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.combine_hid_out_dims,
                    nhead=combiner_heads,
                    batch_first=True,
                ),
                num_layers=combiner_layers,
            )

        if relevant_object_box_uuid is not None and accurate_object_box_uuid is not None:
            num_boxes = 2
            num_cameras = 1
            self.len_bounding_boxes = num_boxes * 5 * num_cameras
            self.bbox_pos_encoder = nn.Sequential(
                PositionalEncoder(32),
                nn.Linear(32, self.combine_hid_out_dims),
                nn.LayerNorm(self.combine_hid_out_dims),
                nn.ReLU(),
            )
            self.coord_pos_enc = nn.Embedding(self.len_bounding_boxes, self.combine_hid_out_dims)

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return self.combine_hid_out_dims

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_dino(self, observations):
        return self.dino_compressor(observations[self.dino_uuid])

    def distribute_target(self, observations):
        max_len = observations[self.goal_uuid].shape[-1]
        goals_tensor = observations[self.goal_uuid].cpu().numpy().astype(np.uint8)
        goals = []
        for g in goals_tensor:
            g = convert_byte_to_string(g, max_len=max_len)
            goals.append(g)
        with torch.no_grad():
            goal_emb = self.text_tokenizer(goals, return_tensors="pt", padding=True).to(
                observations[self.goal_uuid].device
            )
            goal_emb = self.text_goal_encoder(**goal_emb).last_hidden_state
        goal_emb_after_adapter = self.text_adapter(goal_emb)
        return goal_emb_after_adapter

    def encode_bbox(self, observations):
        task_relevant_object_bbox = observations[self.relevant_object_box_uuid]
        nav_accurate_object_bbox = observations[self.accurate_object_box_uuid]
        best_nav_boxes = get_best_of_two_bboxes(task_relevant_object_bbox, nav_accurate_object_bbox)
        B, T, N = best_nav_boxes.shape
        combined_boxes = best_nav_boxes.reshape(B * T, N)
        pos_encoded_boxes = self.bbox_pos_encoder(combined_boxes)
        pos_encoded_boxes = pos_encoded_boxes + self.coord_pos_enc(
            torch.tensor(
                [[i for i in range((self.len_bounding_boxes))]],
                device=pos_encoded_boxes.device,
            ).tile(B * T, 1)
        )
        return pos_encoded_boxes

    def adapt_input(self, observations):
        observations = {**observations}
        dino = observations[self.dino_uuid]
        if self.goal_uuid is not None:
            goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(dino.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = dino.shape[:3]
        else:
            nstep, nsampler = dino.shape[:2]

        observations[self.dino_uuid] = dino.view(-1, *dino.shape[-3:])
        if self.goal_uuid is not None:
            observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(observations)

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        embs = [
            self.fusion_token.view(1, 1, -1).expand(nstep * nsampler, -1, -1),
            self.compress_dino(observations).flatten(start_dim=2).permute(0, 2, 1),
        ]
        if self.goal_uuid is not None:
            text_feats = self.distribute_target(observations)
            embs.append(text_feats)
        if self.relevant_object_box_uuid is not None and self.accurate_object_box_uuid is not None:
            pos_encoded_boxes = self.encode_bbox(observations)
            embs.append(pos_encoded_boxes)
        embs = torch.cat(embs, dim=1)
        x = self.target_obs_combiner(embs)
        x = x[:, 0]

        if self.goal_uuid is None:
            return self.adapt_output(x, use_agent, nstep, nsampler, nagent), None
        else:
            return self.adapt_output(x, use_agent, nstep, nsampler, nagent), self.adapt_output(
                text_feats.mean(dim=1), use_agent, nstep, nsampler, nagent
            )
