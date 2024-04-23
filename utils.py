import jax.numpy as jnp
from flax import linen as nn
from minesweeper import MinesweeperGame
from IPython import display
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

def render_frame(env, action_scores):
    dim_0, dim_1 = len(env.visible), len(env.visible[0])
    dim_factor = dim_0*dim_1
    fig_0, fig_1 = plt.rcParams['figure.figsize']
    fig_factor = fig_0*fig_1
    fontsize=20*math.sqrt(fig_factor)/math.sqrt(dim_factor)
    heat_cmap  = matplotlib.colormaps["RdYlGn"]
    indicator_cmap = matplotlib.colormaps["Dark2"]
    fig, ax = plt.subplots(layout="tight")
    ax.imshow(env.visible, cmap="Greys_r")
    plt.xticks(range(len(env.visible[0])))
    plt.yticks(range(len(env.visible)))
    ax.set_xticks([x - 0.5 for x in ax.get_xticks()][1:], minor='true')
    ax.set_yticks([y - 0.5 for y in ax.get_yticks()][1:], minor='true')
    plt.grid(which='minor')
    for y in range(len(env.player_map)):
        for x in range(len(env.player_map[y])):
          element = env.player_map[y][x]
          if element == "*":
              ax.text(x, y, f"{element}", ha="center", va="center", color="k", fontsize=fontsize*2)
          elif element != " " and element != 0:
              ax.text(x, y, f"{element}", ha="center", va="center", color=indicator_cmap((element-1)/7), fontsize=fontsize)
    x_points = []
    y_points = []
    c_points = []
    for loc, score in action_scores.items():
        y, x = loc
        y_points.append(y)
        x_points.append(x)
        c_points.append(score)
    if c_points:
        min_c, max_c = min(c_points), max(c_points)
        c_points = [(point-min_c)/(max_c-min_c) for point in c_points]
    # ax.scatter(x_points, y_points, c=c_points, cmap="RdYlGn", s=(2400/dim_factor)**2, marker="s")
    # ax.scatter(x_points, y_points, c=c_points, cmap="RdYlGn", marker="s")

    for x, y, c in zip(x_points, y_points, c_points):
        text = ax.text(x, y, f'{round(c*100)}%', ha="center", va="center", color=heat_cmap(c), fontsize=fontsize)#k
    fig.canvas.draw()
    frame_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_rgb = frame_rgb.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    plt.close(fig)
    return frame_rgb

def neural_network_agent(game_map, network_apply, params, enforce_reachability=True, revealed_spaces=False):
    env = MinesweeperGame(game_map, enforce_reachability=enforce_reachability)
    obs, score, done = env.reset()
    steps = 0
    frames = []
    while not done:
      # evaluate network with provided parameters for each action
      action, best = None, None
      action_scores = {}
      for action_idx, action_obs in enumerate(obs):
          action_score = network_apply(params, action_obs.flatten())[0]
          y, x = env._action_keys[action_idx]
          action_scores[(y,x)] = action_score
          # select action with highest score
          if action == None or action_score > best:
              action = action_idx
              best = action_score

      frames.append(render_frame(env, action_scores))
      obs, score, done = env.step(action)
      steps += 1

    frames.append(render_frame(env, {}))
    if revealed_spaces:
        score = env.revealed_spaces
    return score, env.player_map, steps, frames

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

def display_frame(frames, frame_idx):
    plt.figure(layout="tight")
    plt.imshow(frames[frame_idx])
    plt.axis('off')
    plt.show()

def render_full_game(frames):
    interact(display_frame, frames=fixed(frames), frame_idx=widgets.IntSlider(min=0, max=len(frames)-1, step=1))