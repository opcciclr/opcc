import os

import gym
import opcc
import pickle
from pathlib import Path

MAZE_INFO = {
    'd4rl:maze2d-open-dense-v0': "#######\\"
                                 "#OOOOO#\\"
                                 "#OOGOO#\\"
                                 "#OOOOO#\\"
                                 "#######",

    'd4rl:maze2d-umaze-dense-v1': "#####\\"
                                  "#GOO#\\"
                                  "###O#\\"
                                  "#OOO#\\"
                                  "#####",

    'd4rl:maze2d-medium-dense-v1': "########\\"
                                   "#OO##OO#\\"
                                   "#OO#OOO#\\"
                                   "##OOO###\\"
                                   "#OO#OOO#\\"
                                   "#O#OO#O#\\"
                                   "#OOO#OG#\\"
                                   "########",

    'd4rl:maze2d-large-dense-v1': "############\\"
                                  "#OOOO#OOOOO#\\"
                                  "#O##O#O#O#O#\\"
                                  "#OOOOOO#OOO#\\"
                                  "#O####O###O#\\"
                                  "#OO#O#OOOOO#\\"
                                  "##O#O#O#O###\\"
                                  "#OO#OOO#OGO#\\"
                                  "############",
}


def mujoco_dataset_dict(env_name):
    dataset_info = {}
    for dataset_name in ['random', 'expert', 'medium',
                         'medium-replay', 'medium-expert']:
        for split_name, split in [("", None), ("-1k", 1e3),
                                  ("-10k", 1e4), ("-100k", 1e5)]:
            dataset_info[dataset_name + split_name] = \
                {'name': f"d4rl:{env_name.lower()}-{dataset_name}-v2",
                 "split": int(split) if split is not None else None}
    return dataset_info


if __name__ == '__main__':

    maze2d_data, other_data, mujoco_data = {}, {}, {}
    for env_name in os.listdir(opcc.ASSETS_DIR):
        env = gym.make(env_name)
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        assert all(env.action_space.high == max_action)
        env.close()

        # policy-stats
        policy_stats_path = os.path.join(opcc.ASSETS_DIR,
                                         env_name, 'stats.p')
        if os.path.exists(policy_stats_path):
            print('Using Existing stats from :{}'.format(policy_stats_path))
            policy_stats = pickle.load(open(policy_stats_path,
                                            'rb'))['policies']
        else:
            policy_stats = None

        if 'maze' in env_name.lower():
            maze2d_data[env_name] = {
                'observation_size': obs_size,
                'action_size': action_size,
                'max_action': max_action,
                'maze': MAZE_INFO[env_name],
                'datasets': {'1k': {'name': env_name,
                                    'split': 1000},
                             '10k': {'name': env_name,
                                     'split': 10000},
                             '100k': {'name': env_name,
                                      'split': 100000},
                             '1m': {'name': env_name,
                                    'split': 1000000}},
                'policies': policy_stats
            }

        elif env_name in ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2']:
            mujoco_data[env_name] = {
                'observation_size': obs_size,
                'action_size': action_size,
                'max_action': max_action,
                'datasets': mujoco_dataset_dict(env_name.split("-")[0]),
                'policies': policy_stats
            }
        else:
            pass

    print("===================================")
    print(maze2d_data)
    print(mujoco_data)
    print(other_data)

    with open(os.path.join(Path(opcc.ASSETS_DIR).parent.absolute(),
                           'config.py'),
              'w') as config_file:
        config_file.write(f"MAZE_ENV_CONFIGS = {maze2d_data} \n")
        config_file.write(f"MUJOCO_ENV_CONFIGS = {mujoco_data} \n")
        config_file.write(f"OTHER_CONFIGS = {other_data} \n")
        config_file.write("ENV_CONFIGS = {**MAZE_ENV_CONFIGS,"
                          " **MUJOCO_ENV_CONFIGS, **OTHER_CONFIGS}")
