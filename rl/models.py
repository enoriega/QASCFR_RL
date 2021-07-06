from typing import Tuple, List

import torch
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from torch import nn, FloatTensor
# noinspection PyPep8Naming
from torch.nn import functional as F


# noinspection PyAbstractClass,PyPep8Naming
class FeedFwdFocusedReading(nn.Module):

    emb_size = 300

    def __init__(self, input_shape: int, action_space_size: int, use_embeddings: bool, entity_dropout: float,
                 layers_size: List[int]):
        super(FeedFwdFocusedReading, self).__init__()
        # [2100, 1000, 250, 100]
        assert len(layers_size) == 4, "There must be four sizes in layers_size"
        # Flag to determine whether the model will receive embeddings as part of the input
        self._use_embeddings = use_embeddings
        self._entity_dropout = entity_dropout

        # Put a placeholder parameter here
        # self.ff = nn.Sequential(
        #     nn.Linear(input_shape, 500),
        #     nn.Tanh(),
        #     nn.Linear(500, 300),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(300, 100),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(100, 10),
        #     nn.Tanh()
        # )
        self.ff = nn.Sequential(
            nn.Linear(input_shape, layers_size[0]),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(layers_size[0], layers_size[1]),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(layers_size[1], layers_size[2]),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(layers_size[2], layers_size[3]),
            nn.Tanh()
        )
        self.pi = nn.Linear(layers_size[3], action_space_size)
        self.v = nn.Linear(layers_size[3], 1)

    # noinspection PyUnusedLocal
    def forward(self, obs: FloatTensor,
                prev_action: FloatTensor,
                prev_reward: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        Does a forward pass through the network
        :param obs: Current state's observation tensor
        :param prev_action: Previous action as a one-hot vector
        :param prev_reward: Previous reward as a one-element vector
        :return: Weights of the categorical distribution of the actions, the estimated state value
        """

        # Infer the batch and time step dimensions
        lead_dim, T, B, shape = infer_leading_dims(obs, 1)

        obs = obs.view(T * B, *shape)

        # Split the embeddings from the other features
        if self._use_embeddings:
            embs = obs[:, : FeedFwdFocusedReading.emb_size * 2]
            features = obs[:, FeedFwdFocusedReading.emb_size * 2:]

            # Put a dropout layer on over the embeddings, with default p
            embs = F.dropout(embs, p=self._entity_dropout)

            # Reassemble the input after adding dropout to the embeddings
            x = torch.cat([embs, features], dim=1)
        else:
            x = obs

        x = self.ff(x)

        dist = self.pi(x)
        dist = F.softmax(dist, -1)

        val = self.v(x).squeeze(-1)

        dist, val = restore_leading_dims((dist, val), lead_dim, T, B)
        return dist, val


class FFFRMedium(FeedFwdFocusedReading):
    def __init__(self, input_shape: int, action_space_size: int, use_embeddings: bool, entity_dropout: float):
        super(FFFRMedium, self).__init__(input_shape, action_space_size, use_embeddings, entity_dropout, [500, 300, 100, 10])


class FFFRLarge(FeedFwdFocusedReading):
    def __init__(self, input_shape: int, action_space_size: int, use_embeddings: bool, entity_dropout: float):
        super(FFFRLarge, self).__init__(input_shape, action_space_size, use_embeddings, entity_dropout, [2100, 1000, 250, 100])


class FFFRExtraLarge(FeedFwdFocusedReading):
    def __init__(self, input_shape: int, action_space_size: int, use_embeddings: bool, entity_dropout: float):
        super(FFFRExtraLarge, self).__init__(input_shape, action_space_size, use_embeddings, entity_dropout, [4200, 2000, 500, 200])