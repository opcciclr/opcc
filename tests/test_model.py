import gym
import pytest
import torch

import opcc


@pytest.mark.parametrize('env_name, policy_id',
                         [(env_name, policy_id)
                          for env_name in ['d4rl:maze2d-open-dense-v0',
                                           'd4rl:maze2d-umaze-dense-v1',
                                           'd4rl:maze2d-medium-dense-v1',
                                           'd4rl:maze2d-large-dense-v1', ]
                          for policy_id in range(1, 6)]
                         + [(env_name, policy_id)
                            for env_name in ['HalfCheetah-v2',
                                             'Hopper-v2',
                                             'Walker2d-v2']
                            for policy_id in range(1, 11)])
def test_model_exists(env_name, policy_id):
    model, model_info = opcc.get_policy(env_name, policy_id)
    for key in ['score_mean', 'score_std']:
        assert key in model_info, ('{} not found in model_info of '
                                   'environment {} and policy-id {}'.
                                   format(env_name, policy_id))
        assert (isinstance(model_info[key], float)
                or isinstance(model_info[key], int)), \
            ('value of {} is not a number for environment {}'
             ' and policy-id {}'.format(key, env_name, policy_id))

    env = gym.make(env_name)
    obs = env.reset()
    obs = torch.tensor(obs).unsqueeze(0)
    obs = obs.repeat(10, 1)  # just making a bigger batch
    action_batch = model(obs)
    assert action_batch.shape == (10, env.action_space.shape[0])

    env.step(action_batch[0].data.cpu().numpy().astype('float32'))
    env.close()
