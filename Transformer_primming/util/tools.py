import random
import os
import numpy as np
import torch
from torch import nn
from transformers.activations import ACT2FN

class NeuralConfig:
    """ Configuration for neural network components. """
    tag_count = 10
    prioritize_batch = False
    normalization_epsilon = 1e-12
    activation_function = 'gelu'
    dropout_rate = 0.1
    input_dimension = 768
    output_dimension = 768

neural_config = NeuralConfig()

def set_all_seeds(seed_value=1029):
    """ Set all seeds to make all operations deterministic. """
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ConditionalRandomField(nn.Module):
    """ Implementation of a Conditional Random Field (CRF). """
    def __init__(self, tags: int, batch_first_flag: bool = False):
        super().__init__()
        if tags <= 0:
            raise ValueError("Number of tags must be positive.")
        self.tags = tags
        self.batch_first_flag = batch_first_flag
        self.initial_transitions = nn.Parameter(torch.empty(tags))
        self.final_transitions = nn.Parameter(torch.empty(tags))
        self.inter_tag_transitions = nn.Parameter(torch.empty(tags, tags))
        self.initialize_parameters()

    def initialize_parameters(self):
        """ Initializes CRF parameters uniformly. """
        nn.init.uniform_(self.initial_transitions, -0.1, 0.1)
        nn.init.uniform_(self.final_transitions, -0.1, 0.1)
        nn.init.uniform_(self.inter_tag_transitions, -0.1, 0.1)

    def forward(self, emission_scores: torch.Tensor, tag_sequence: torch.LongTensor,
                sequence_mask: Optional[torch.ByteTensor] = None, reduction_method: str = 'mean') -> torch.Tensor:
        """ Compute log likelihood of a sequence of tags given the emissions. """
        self.validate_inputs(emission_scores, tag_sequence, sequence_mask)
        if self.batch_first_flag:
            emission_scores, tag_sequence, sequence_mask = (
                emission_scores.transpose(0, 1), tag_sequence.transpose(0, 1), sequence_mask.transpose(0, 1))
        log_likelihood = self.compute_log_likelihood(emission_scores, tag_sequence, sequence_mask)
        return self.apply_reduction(log_likelihood, reduction_method)

    def decode(self, emission_scores: torch.Tensor, sequence_mask: Optional[torch.ByteTensor] = None,
               best_paths_count: Optional[int] = None, padding_tag: Optional[int] = None) -> List[List[List[int]]]:
        """ Find the most likely tag sequence using the Viterbi algorithm. """
        if best_paths_count is None:
            best_paths_count = 1
        if sequence_mask is None:
            sequence_mask = torch.ones(emission_scores.shape[:2], dtype=torch.uint8, device=emission_scores.device)
        self.validate_inputs(emission_scores, mask=sequence_mask)

        if self.batch_first_flag:
            emission_scores, sequence_mask = emission_scores.transpose(0, 1), sequence_mask.transpose(0, 1)

        return self.viterbi_decode(emission_scores, sequence_mask, best_paths_count, padding_tag)

    def validate_inputs(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor] = None,
                        mask: Optional[torch.ByteTensor] = None) -> None:
        """ Validate input parameters for consistency. """
        if emissions.dim() != 3:
            raise ValueError(f'Emissions must be 3-dimensional, got {emissions.dim()}')
        if emissions.size(2) != self.tags:
            raise ValueError(f'Expected last dimension of emissions to be {self.tags}, got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError('First two dimensions of emissions and tags must match.')
        if mask is not None and emissions.shape[:2] != mask.shape:
            raise ValueError('First two dimensions of emissions and mask must match.')

class FullyConnectedLayer(nn.Module):
    def __init__(self, config: NeuralConfig, dropout_prob: float):
        super(FullyConnectedLayer, self).__init__()
        self.dense_layer = nn.Linear(config.input_dimension, config.output_dimension)
        self.normalization_layer = nn.LayerNorm(config.output_dimension, eps=config.normalization_epsilon)
        self.activation = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the layer. """
        output = self.dense_layer(inputs)
        output = self.activation(output)
        output = self.normalization_layer(output)
        return self.dropout(output)
