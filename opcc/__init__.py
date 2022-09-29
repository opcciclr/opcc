import os
import pickle
from pathlib import Path
import d4rl
import gym
import torch
import hashlib
import numpy as np

from d4rl.pointmaze import waypoint_controller

ASSETS_DIR = os.path.join(Path(os.path.dirname(__file__)), "assets")


class Maze2dController:
    def __init__(self, maze, target):
        self.controller = waypoint_controller.WaypointController(maze)
        self.__target = target

    def forward(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self

    def _get_action(self, obs):
        if len(obs.shape) == 1:
            position = obs[0:2]
            velocity = obs[2:4]
            action, _ = self.controller.get_action(position,
                                                   velocity,
                                                   self.__target)
            return action
        elif len(obs.shape) > 1:
            action_batch = []
            for _obs in obs:
                action_batch.append(self(_obs))
            return action_batch

    def __call__(self, obs):
        is_tensor = False
        if torch.is_tensor(obs):
            is_tensor = True
            device = obs.device
            obs = obs.cpu().data.numpy()

        action = np.array(self._get_action(obs))
        if is_tensor:
            action = torch.tensor(action).double().to(device)

        return action


def get_queries(env_name):
    """
    Retrieves queries for the environment.

    :param env_name:  name of the environment

    Example:
        >>> import opcc
        >>> opcc.get_queries('Hopper-v2')
    """
    from .config import ENV_CONFIGS
    assert env_name in ENV_CONFIGS, \
        ('`{}` not found. It should be among following: {}'
         .format(env_name, list(ENV_CONFIGS.keys())))

    env_dir = os.path.join(ASSETS_DIR, env_name)
    query_dir = os.path.join(env_dir, 'queries')
    query_chunks_file_paths = sorted([os.path.join(query_dir, x) for x in
                                      os.listdir(query_dir)
                                      if 'queries.' in x])
    query_bytes = b""
    for file_path in query_chunks_file_paths:
        with open(file_path, 'rb') as query_chunk:
            query_bytes += query_chunk.read()

    with open(os.path.join(query_dir, 'md5sums.txt'), 'r') as checksum_file:
        original_check_sum = checksum_file.read()

    if hashlib.md5(query_bytes).hexdigest() \
            != original_check_sum.split(" ")[0]:
        raise IOError(f'queries in {query_dir} are corrupted!')

    queries = pickle.loads(query_bytes)
    return queries


def get_policy(env_name: str, id: int = 1):
    """
    Retrieves policies for the environment with the pre-trained quality marker.

    :param env_name:  name of the environment
    :param id: Id of the policy.

    Example:
        >>> import opcc
        >>> opcc.get_policy('d4rl:maze2d-open-dense-v0',id=1)
    """
    from .config import ENV_CONFIGS

    assert env_name in ENV_CONFIGS, \
        ('{} is invalid. Expected values include {}'
         .format(env_name, ENV_CONFIGS.keys()))

    assert id in ENV_CONFIGS[env_name]['policies'].keys(), \
        ('id should be among {}'.format(ENV_CONFIGS[env_name]
                                        ['policies'].keys()))

    if 'maze2d' in env_name:
        maze = ENV_CONFIGS[env_name]['maze']
        target = ENV_CONFIGS[env_name]['policies'][id]['target']
        model = Maze2dController(maze, target)
    else:
        # retrieve model
        model_dir = os.path.join(ASSETS_DIR, env_name, 'policies',
                                 'policy_{}'.format(id))
        model_path = os.path.join(model_dir, 'model.p')
        assert os.path.exists(model_path), \
            'model not found @ {}'.format(model_path)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # create model
        from .model import Actor
        model = Actor(ENV_CONFIGS[env_name]['observation_size'],
                      ENV_CONFIGS[env_name]['action_size'],
                      ENV_CONFIGS[env_name]['max_action'])
        model.load_state_dict(state_dict['actor'])

        # Note: Gym returns observations with numpy float64( or double) type.
        # And, if the model is in "float" ( or float32) then we need to
        # downcast the observation to float32 before feeding them to the
        # network. However, this down-casting leads to miniscule differences
        # in precision over different system (processors). Though, these
        # differences are miniscule, they get propagated to the predicted
        # actions which over longer horizons which when feedback back to the
        # gym-environment lead to small but significant difference in
        # trajectories as reflected in monte-carlo return.

        # In order to prevent above scenario, we simply upcast our model to
        # double.
        model = model.double()

    info = {'score_mean': ENV_CONFIGS[env_name]['policies'][id]['score_mean'],
            'score_std': ENV_CONFIGS[env_name]['policies'][id]['score_std']}
    return model, info


def get_sequence_dataset(env_name, dataset_name):
    from .config import ENV_CONFIGS

    assert env_name in ENV_CONFIGS, \
        ('{} is invalid. Expected values include {}'
         .format(env_name, ENV_CONFIGS.keys()))
    assert dataset_name in ENV_CONFIGS[env_name]['datasets'], \
        ('`{}` not found. It should be among following: {}'.
         format(dataset_name, list(ENV_CONFIGS[env_name]['datasets'].keys())))

    dataset_env = ENV_CONFIGS[env_name]['datasets'][dataset_name]['name']
    env = gym.make(dataset_env)
    dataset = env.get_dataset()
    # remove meta-data as the sequence dataset doesn't work with it.
    metadata_keys = [k for k in dataset.keys() if 'meta' in k]
    for k in metadata_keys:
        dataset.pop(k)

    split = ENV_CONFIGS[env_name]['datasets'][dataset_name]['split']
    if split is not None:
        dataset = {k: v[:split] for k, v in dataset.items()}

    dataset = [x for x in d4rl.sequence_dataset(env, dataset)]
    return dataset


def get_qlearning_dataset(env_name, dataset_name):
    from .config import ENV_CONFIGS
    assert env_name in ENV_CONFIGS, \
        ('{} is invalid. Expected values include {}'
         .format(env_name, ENV_CONFIGS.keys()))
    assert dataset_name in ENV_CONFIGS[env_name]['datasets'], \
        ('`{}` not found. It should be among following: {}'.
         format(dataset_name, list(ENV_CONFIGS[env_name]['datasets'].keys())))

    dataset_env = ENV_CONFIGS[env_name]['datasets'][dataset_name]['name']
    env = gym.make(dataset_env)
    dataset = d4rl.qlearning_dataset(env)

    split = ENV_CONFIGS[env_name]['datasets'][dataset_name]['split']
    if split is not None:
        dataset = {k: v[:split] for k, v in dataset.items()}
    return dataset


def get_dataset_names(env_name):
    from .config import ENV_CONFIGS
    assert env_name in ENV_CONFIGS, \
        ('`{}` not found. It should be among following: {}'.
         format(env_name, list(ENV_CONFIGS.keys())))
    return list(ENV_CONFIGS[env_name]['datasets'].keys())


def load_query_dataset_distance(env_name, dataset_name):
    from .config import ENV_CONFIGS
    assert env_name in ENV_CONFIGS, \
        ('{} is invalid. Expected values include {}'
         .format(env_name, ENV_CONFIGS.keys()))
    assert dataset_name in ENV_CONFIGS[env_name]['datasets'], \
        ('`{}` not found. It should be among following: {}'.
         format(dataset_name, list(ENV_CONFIGS[env_name]['datasets'].keys())))

    env_dir = os.path.join(ASSETS_DIR, env_name)
    query_dir = os.path.join(env_dir, 'queries')
    distances_dir = os.path.join(env_dir, 'distances', dataset_name)
    distances_chunks_file_paths = sorted([os.path.join(distances_dir, x)
                                          for x in os.listdir(distances_dir)
                                          if 'distances.' in x])
    distances_bytes = b""
    for file_path in distances_chunks_file_paths:
        with open(file_path, 'rb') as query_chunk:
            distances_bytes += query_chunk.read()

    # ######################################
    # ensure distances belong to the queries
    # ######################################
    with open(os.path.join(query_dir, 'md5sums.txt'), 'r') as checksum_file:
        query_check_sum = checksum_file.read()

    with open(os.path.join(query_dir, 'queries_md5sums.txt'), 'r')\
            as checksum_file:
        relative_query_check_sum = checksum_file.read()

    if query_check_sum != relative_query_check_sum:
        raise IOError(f'estimated distances don\'t belong to queries saved '
                      f'in {query_dir}! \n Checksums don\'t match')

    # ##############################################
    # Validate distances retrieved are not corrupted
    # ##############################################
    with open(os.path.join(distances_dir, 'distances_md5sums.txt'),
              'r') as checksum_file:
        original_distances_check_sum = checksum_file.read()

    if hashlib.md5(distances_bytes).hexdigest() \
            != original_distances_check_sum.split(" ")[0]:
        raise IOError(f'distances in {query_dir}! \n are corrupted!')

    distances = pickle.loads(distances_bytes)

    return distances
