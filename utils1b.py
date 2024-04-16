import jax.numpy as jnp
from flax import linen as nn
from minesweeper import MinesweeperGame

def agent_encoder(agent):
    temp = {'params': dict()}
    for layer_key in agent['params']:
        temp['params'][layer_key] = dict()
        for param_key, params in agent['params'][layer_key].items():
            if len(params.shape) == 1:
                temp['params'][layer_key][param_key] = [float(num) for num in params]
            elif len(params.shape) == 2:
                temp['params'][layer_key][param_key] = [[float(num) for num in seq] for seq in params]
    return temp

def agent_decoder(agent):
    temp = {'params': dict()}
    for layer_key in agent['params']:
        temp['params'][layer_key] = dict()
        for param_key, params in agent['params'][layer_key].items():
            temp['params'][layer_key][param_key] = jnp.array(params)
    return temp

class MLP(nn.Module):
    """Simple ReLU MLP"""

    num_hidden_units: int
    num_hidden_layers: int
    num_output_units: int

    @nn.compact
    def __call__(self, x):
        for l in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.num_output_units)(x)
        return x

def neural_network_agent(game_map, network_apply, params, enforce_reachability=True, revealed_spaces=False):
    env = MinesweeperGame(game_map, enforce_reachability=enforce_reachability)
    obs, score, done = env.reset()
    steps = 0
    while not done:
        # evaluate network with provided parameters for each action
        action, best = None, None
        for action_idx, action_obs in enumerate(obs):
            action_score = network_apply(params, action_obs.flatten())[0]
            # select action with highest score
            if action == None or action_score > best:
                action = action_idx
                best = action_score
        obs, score, done = env.step(action)
        steps += 1
    if revealed_spaces:
        score = env.revealed_spaces
    return score, env.player_map, steps

def map_check(game_instance):
    if isinstance(game_instance, list|tuple):
        valid = True
        for seq in game_instance:
            valid = valid and isinstance(seq, list|tuple)
    assert valid, "Problem instance should be a list or tuple"

    for seq in game_instance:
        for elem in seq:
            valid = valid and isinstance(elem, int) and (elem == 1 or elem == 0)
    assert valid, "Problem instance must contain only integers 1 or 0"

    size = len(game_instance[-1])
    for seq in game_instance:
        valid = valid and len(seq) == size
    assert valid, "Problem instance must have consistent-sized rows and columns"

    dim0, dim1 = len(game_instance), len(game_instance[-1])
    starting_mine = game_instance[dim0//2][dim1//2] == 1
    mine_count = sum([sum(seq) for seq in game_instance])

    return starting_mine, mine_count
