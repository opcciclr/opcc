# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from pathlib import Path

import gym
import numpy as np
import torch
from tqdm import tqdm

import opcc
from opcc.config import ENV_CONFIGS


def generate_env_stats(env_name,
                       test_episodes,
                       stats_dir,
                       no_cache=False):
    env_dir = os.path.join(stats_dir, env_name)
    env_policies_dir = os.path.join(env_dir, 'policies')
    env_stats_path = os.path.join(env_dir, 'stats.p')
    if not no_cache:
        if os.path.exists(env_stats_path):
            print('Using Existing stats from :{}'.format(env_stats_path))
            return pickle.load(open(env_stats_path, 'rb'))

    env_config = ENV_CONFIGS[env_name]
    for policy_id in tqdm(sorted(env_config['policies'].keys())):
        policy, _ = opcc.get_policy(env_name, policy_id)
        episode_rewards = []
        obs_images = []
        for episode_i in range(test_episodes):
            env = gym.make(env_name)
            env.seed(episode_i)
            done = False
            episode_reward = 0
            obs = env.reset()
            while not done:
                if episode_i == 0:
                    # env.render()
                    obs_images.append(env.render(mode='rgb_array'))
                action = policy(torch.tensor(obs).unsqueeze(0))
                action = action.data.numpy()[0].astype('float32')
                obs, reward, done, step_info = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
            if episode_i == 0:
                obs_images.append(env.render(mode='rgb_array'))
            env.close()

        episode_rewards = np.array(episode_rewards)
        mean = round(episode_rewards.mean(), 2)
        std = round(episode_rewards.std(), 2)
        env_config['policies'][policy_id]['score_mean'] = mean
        env_config['policies'][policy_id]['score_std'] = std

        # obs_images = np.moveaxis(obs_images, -1, 1)
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(list(obs_images), fps=32)

        policy_dir_path = os.path.join(env_policies_dir,
                                       'policy_{}'.format(policy_id))
        os.makedirs(policy_dir_path, exist_ok=True)
        clip.write_gif(os.path.join(policy_dir_path, 'episode.gif'),
                       fps=32)
    pickle.dump(env_config, open(env_stats_path, 'wb'))
    return env_config


def markdown_pre_trained_scores(env_info):
    # create markdown for the table:
    import math
    msg = ""
    for i, env_name in enumerate(env_info):
        policy_ids_set = sorted(env_info[env_name]['policies'].keys())
        msg += "{}. `{}` \n".format(i, env_name)
        for policy_ids in [policy_ids_set[_*5:(_*5)+5]
                           for _ in range(math.ceil(len(policy_ids_set)/5))]:
            # policy-ids
            msg += "|`ID`"
            for policy_id in policy_ids:
                msg += "|" + "`{}`".format(policy_id)
            msg += " | \n"

            msg += "|" + " | ".join(":------:" for _ in range(len(policy_ids) + 1
                                                              )) + ' | ' + '\n'
            # scores
            msg += "|__Performance__"
            for policy_id in policy_ids:
                msg += '| {}±{} <br> '.format(env_info[env_name]['policies']
                                        [policy_id]['score_mean'],
                                        env_info[env_name]['policies']
                                        [policy_id]['score_std'])
            msg += " | \n"

            # scores
            msg += "|__Render__"
            for policy_id in policy_ids:
                msg += '| {}'.format('![](opcc/assets/{}/policies/'
                                     'policy_{}/episode.gif)'.format(env_name,
                                                                     policy_id))
            msg += " | \n\n"
    return msg


if __name__ == '__main__':
    # Let's gather arguments
    parser = argparse.ArgumentParser(description='Generate stats for '
                                                 'environment')
    parser.add_argument('--env-name', required=False, type=str,
                        help='Name of the environment',
                        default='d4rl:maze2d-open-dense-v0')
    parser.add_argument('--test-episodes', required=False, default=20,
                        type=int, help='No. of episodes for evaluation')
    parser.add_argument('--all-envs', default=False, action='store_true',
                        help="Generate stats for all envs "
                             "(default: %(default)s)")
    parser.add_argument('--no-cache', default=False, action='store_true',
                        help="Doesn't use pre-generated stats "
                             " (default: %(default)s)")
    parser.add_argument('--stats-dir', type=str,
                        default=os.path.join(str(Path.home()), '.opcc',
                                             'generated_stats'))
    parser.add_argument('--render', default=False, action='store_true',
                        help="Renders the environment while evaluating "
                             " (default: %(default)s)")
    args = parser.parse_args()
    os.makedirs(args.stats_dir, exist_ok=True)
    stats_info = {}

    for env_name in tqdm(ENV_CONFIGS.keys()
                         if args.all_envs else [args.env_name]):
        stats_info[env_name] = generate_env_stats(env_name,
                                                  args.test_episodes,
                                                  args.stats_dir,
                                                  no_cache=args.no_cache)
    print(stats_info)
    table_markdown = markdown_pre_trained_scores(stats_info)
    print(table_markdown)
