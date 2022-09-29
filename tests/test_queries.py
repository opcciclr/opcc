import os

import gym
import numpy as np
import opcc
import pytest
import torch
from opcc.config import ENV_CONFIGS

ENV_LISTS = ['d4rl:maze2d-open-dense-v0',
             'd4rl:maze2d-umaze-dense-v1',
             'd4rl:maze2d-medium-dense-v1',
             'd4rl:maze2d-large-dense-v1',
             'Hopper-v2',
             'HalfCheetah-v2',
             'Walker2d-v2']


def mc_return(env_name, sim_states, open_loop_actions, policy_horizon, policy,
              runs):
    all_returns, all_steps = [], []
    for sim_state_i, sim_state in enumerate(sim_states):
        envs = []
        for _ in range(runs):
            env = gym.make(env_name)
            env.reset()
            env.sim.set_state_from_flattened(
                np.array(sim_state).astype("float64"))
            env.sim.forward()
            envs.append(env)

        obss = [None for _ in range(runs)]
        dones = [False for _ in range(runs)]
        returns = [0 for _ in range(runs)]
        steps = [0 for _ in range(runs)]

        open_loop_horizon = open_loop_actions.shape[1]
        for step_i in range(open_loop_horizon + policy_horizon):
            for env_i, env in enumerate(envs):
                if not dones[env_i]:
                    if step_i < open_loop_horizon:
                        step_action = open_loop_actions[sim_state_i, step_i]
                        step_action = np.array(step_action)

                    else:
                        with torch.no_grad():
                            obs = torch.tensor(obss[env_i]).unsqueeze(0)
                            step_action = policy.actor(obs).data.cpu().numpy()
                            step_action = step_action[0]

                    step_action = step_action.astype("float32")
                    obs, reward, done, info = env.step(step_action)

                    obss[env_i] = obs
                    dones[env_i] = done or dones[env_i]
                    returns[env_i] += reward
                    steps[env_i] += 1

        [env.close() for env in envs]
        all_returns.append(returns)
        all_steps.append(steps)

    return np.array(all_returns).mean(1)


@pytest.mark.parametrize("env_name", ENV_LISTS)
def test_get_queries(env_name):
    # env info
    env = gym.make(env_name)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    env.close()

    # query keys
    keys = ["obs_a", "obs_b", "action_a", "action_b", "policy_horizon",
            "target"]
    info_keys = [
        "return_a",
        "return_b",
        "state_a",
        "state_b",
        "policy_horizon_a",
        "policy_horizon_b",
        "policy_actions_a",
        "policy_actions_b",
        "runs",
    ]

    # iterate through queries
    queries = opcc.get_queries(env_name)
    for (policy_a_id, policy_b_id), query_batch in queries.items():
        assert policy_a_id[
                   0] == env_name, "policy_a_id[0] does not match " "with environment name"
        assert policy_b_id[
                   0] == env_name, "policy_a_id[0] does not match " "with environment name"
        assert isinstance(policy_a_id[1],
                          int), "policy_a_id[1] is not an " "integer"
        assert isinstance(policy_b_id[1],
                          int), "policy_b_id[1] is not an " "integer"

        # get policies
        policy_a, _ = opcc.get_policy(*policy_a_id)
        policy_b, _ = opcc.get_policy(*policy_b_id)

        # validate keys in queries
        for key in keys:
            assert key in query_batch.keys(), "{} not in query_batch".format(
                key)
        for key in info_keys:
            assert key in query_batch[
                "info"].keys(), "{} not in query_batch".format(key)

        # get query data
        obs_a = query_batch["obs_a"]
        obs_b = query_batch["obs_b"]
        action_a = query_batch["action_a"]
        action_b = query_batch["action_b"]
        target = query_batch["target"]

        # ensure all are list
        assert isinstance(action_a, list), "action_a is not a list"
        assert isinstance(action_b, list), "action_b is not a list"
        assert isinstance(obs_a, list), "obs_a is not a list"
        assert isinstance(obs_b, list), "obs_b is not a list"
        assert isinstance(target, list), "target is not a list"

        # check batch size
        query_count = len(obs_a)
        for key in keys:
            assert query_count == len(query_batch[
                                          key]), "Query-count does not match for {} in env. {}" " for ({},{})".format(
                key, env_name, policy_a_id, policy_b_id
            )
        for key in info_keys:
            if key != "runs":
                assert query_count == len(
                    query_batch["info"][key]
                ), "Query-count does not match for {} in env. {}" " for ({},{})".format(
                    key, env_name, policy_a_id, policy_b_id
                )

        # check sub types and lengths
        for i, h in enumerate(query_batch["policy_horizon"]):
            # actions
            assert len(action_a[
                           i]) == action_size, "action-size does not match for action_a in env. {}" " for ({},{})".format(
                env_name, policy_a_id, policy_b_id
            )
            assert isinstance(action_a[i],
                              list), "action_a[{}] is not a list in env. {} for ({},{})".format(
                i, env_name, policy_a_id, policy_b_id
            )

            assert len(action_b[
                           i]) == action_size, "action-size does not match for action_a in env. {}" " for ({},{})".format(
                env_name, policy_a_id, policy_b_id
            )
            assert isinstance(action_b[i],
                              list), "action_a[{}] is not a list in env. {} for ({},{})".format(
                i, env_name, policy_a_id, policy_b_id
            )

            # observations
            assert len(obs_a[
                           i]) == obs_size, "obs-size does not match for obs_a[{}] in env. {}" " for ({},{})".format(
                i, env_name, policy_a_id, policy_b_id
            )
            assert isinstance(obs_a[i],
                              list), "obs_a[{}] is not a list in env. {} for ({},{})".format(
                i, env_name, policy_a_id, policy_b_id
            )

            assert len(obs_b[
                           i]) == obs_size, "obs-size does not match for obs_b[{}] in env. {}" " for ({},{})".format(
                i, env_name, policy_a_id, policy_b_id
            )
            assert isinstance(obs_b[i],
                              list), "obs_b[{}] is not a list in env. {} for ({},{})".format(
                i, env_name, policy_a_id, policy_b_id
            )

            # info
            assert (
                    len(query_batch["info"]["policy_actions_a"][i]) <= h
            ), "policy_actions_a does not match with policy horizon {} in " "query idx {} of env. {} for ({},{})".format(
                h, i, env_name, policy_a_id, policy_b_id
            )

            assert (
                    len(query_batch["info"]["policy_actions_b"][i]) <= h
            ), "policy_actions_b does not match with policy horizon {} in " "query idx {} of env. {} for ({},{})".format(
                h, i, env_name, policy_a_id, policy_b_id
            )

            assert query_batch["info"]["runs"] == len(
                query_batch["info"]["policy_actions_a"][i])
            assert query_batch["info"]["runs"] == len(
                query_batch["info"]["policy_actions_b"][i])

            for run in range(query_batch["info"]["runs"]):
                assert len(
                    query_batch["info"]["policy_actions_a"][i][run]) <= h, (
                    "policy_actions_a does not match with "
                    "policy horizon {} in "
                    "query idx {} of env. {} for ({},{})".format(h, i,
                                                                 env_name,
                                                                 policy_a_id,
                                                                 policy_b_id)
                )

                assert len(
                    query_batch["info"]["policy_actions_b"][i][run]) <= h, (
                    "policy_horizon_b does not match with "
                    "policy horizon {} in "
                    "query idx {} of env. {} for ({},{})".format(h, i,
                                                                 env_name,
                                                                 policy_a_id,
                                                                 policy_b_id)
                )


@pytest.mark.parametrize("env_name", ENV_LISTS)
@pytest.mark.skipif(
    os.environ.get("SKIP_QUERY_TARGET_TESTS", default="0") == "1",
    reason="forcefully skipped by user")
def test_query_targets(env_name):
    queries = opcc.get_queries(env_name)

    for (policy_a_id, policy_b_id), query_batch in queries.items():
        policy_a, policy_b = None, None
        if policy_a_id is not None:
            policy_a, _ = opcc.get_policy(*policy_a_id)
        if policy_b_id is not None:
            policy_b, _ = opcc.get_policy(*policy_b_id)

        target = query_batch["target"]

        open_loop_horizons = query_batch["open_loop_horizon"]
        policy_horizons = query_batch["policy_horizon"]
        horizons = open_loop_horizons + policy_horizons

        for horizon in np.unique(open_loop_horizons + policy_horizons,
                                 return_counts=False):
            _filter = horizons == horizon
            state_a = query_batch["info"]["state_a"][_filter]
            state_b = query_batch["info"]["state_b"][_filter]
            action_a = query_batch["action_a"][_filter]
            action_b = query_batch["action_b"][_filter]
            return_a = mc_return(
                env_name, state_a, action_a, policy_horizons[_filter][0],
                policy_a, query_batch["info"]["runs"]
            )
            return_b = mc_return(
                env_name, state_b, action_b, policy_horizons[_filter][0],
                policy_b, query_batch["info"]["runs"]
            )
            predict = return_a < return_b

            target_return_a = query_batch["info"]["return_a"][_filter]
            target_return_b = query_batch["info"]["return_b"][_filter]

            # Note: Sometimes different processors(or slight variation in
            # numpy version) cause difference in precision leading to  error
            # accumulation over multiple steps. This creates small differences
            # in value estimates. Also, this value differences gets exaggerated
            # with long horizons.
            assert all(
                (return_a - target_return_a) <= 4
            ), "Estimates of Query-A don't match for policies: {} and " "horizon: {}.\n Found: {} Expected: {}".format(
                (policy_a_id, policy_b_id), horizon, return_a, target_return_a
            )

            assert all(
                (return_b - target_return_b) <= 4
            ), "Estimates of Query-B don't match for policies: {} and " "horizon: {}.\n Found: {} Expected: {}".format(
                (policy_a_id, policy_b_id), horizon, return_b, target_return_b
            )

            assert all(target[
                           _filter] == predict), "Query targets do not match for policies: {} and horizon: {}".format(
                (policy_a_id, policy_b_id), horizon
            )
