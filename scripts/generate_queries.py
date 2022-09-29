"""
Usage: python generate_queries.py --env-name d4rl:maze2d-open-v0 --dataset-name 1k --policy-ids 1,2,3,4
"""

import argparse
import heapq
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import gym
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import wandb
from sklearn.decomposition import PCA

import opcc
from kd import KDTree
from opcc import ASSETS_DIR


def mc_return(env, obs, sim_state, init_actions, policy_horizon, policy, max_episodes, estimate_db_distance=None):
    assert len(init_actions) + policy_horizon <= env._max_episode_steps

    sim_state = np.array(sim_state).astype("float64")
    init_action = np.array(init_actions).astype("float32")
    open_loop_horizon = len(init_action)

    expected_score = []
    expected_open_loop_horizon = []
    expected_policy_horizon = []
    db_distances = defaultdict(lambda: 0)
    episodes_actions = []
    episodes_rewards = []
    for ep_i in range(max_episodes):
        env.reset()
        env.sim.set_state_from_flattened(sim_state)
        env.sim.forward()

        score = 0
        step_count = 0
        _open_loop_count = 0
        _policy_count = 0
        done = False

        episode_obs = []
        episode_actions = []
        episode_rewards = []
        while not done and step_count < (open_loop_horizon + policy_horizon):

            if step_count < open_loop_horizon:
                action = init_actions[step_count]
                _open_loop_count += 1
            else:
                with torch.no_grad():
                    obs = torch.tensor(obs).unsqueeze(0)
                    if policy is not None:
                        action = policy(obs).data.cpu().numpy()[0]
                    else:
                        action = env.action_space.sample()
                    obs = obs.cpu().numpy()[0]
                _policy_count += 1

            episode_obs.append(obs)
            episode_actions.append(action.tolist())
            # env.render()
            obs, reward, done, info = env.step(action.astype("float32"))
            episode_rewards.append(reward)
            step_count += 1
            score += reward

        _db_dist = estimate_db_distance(np.concatenate((episode_obs, episode_actions), 1))
        for k in _db_dist:
            db_distances[k] += np.mean([list(x.values())[0] for x in _db_dist[k]])
        expected_score.append(score)
        expected_open_loop_horizon.append(_open_loop_count)
        expected_policy_horizon.append(_policy_count)
        episodes_actions.append(episode_actions[1:])
        episodes_rewards.append(episode_rewards)

    for k in db_distances:
        db_distances[k] /= max_episodes

    return (
        expected_score,
        episodes_actions,
        episodes_rewards,
        expected_open_loop_horizon,
        expected_policy_horizon,
        db_distances,
    )


def generate_candidate_initial_states(
    env, policies, max_transaction_count, save_prob: float, kd_interval: int, action_noise: float
):
    env_states = {"obs": [], "state": []}
    done = True
    step = 0
    first = True

    while len(env_states["state"]) < max_transaction_count:
        if done:
            obs = env.reset()
            policy = random.choice(list(policies.values()))

        if len(env_states["obs"]) < kd_interval:
            if random.random() >= save_prob:
                env_states["obs"].append(obs.tolist())
                env_states["state"].append(env.sim.get_state().flatten().tolist())
        else:
            if len(env_states["obs"]) % kd_interval == 0:
                kd_tree = KDTree(env_states["obs"])
                if first:
                    env_states = {"obs": [], "state": []}
                    first = False
            state = env.sim.get_state().flatten()
            distance = list(kd_tree.get_knn(obs, k=1).values())[0]
            distance_prob_threshold = max(0.9 - 0.00001 * step, 0)
            step += 1
            if distance * 0.1 > distance_prob_threshold:
                env_states["obs"].append(obs.tolist())
                env_states["state"].append(state.tolist())
        action = policy(torch.tensor(obs).unsqueeze(0))
        noise = torch.normal(0, action_noise, size=action.shape)
        step_action = (action + noise).data.cpu().numpy()[0]
        step_action = step_action.astype("float32")
        obs, _, done, _ = env.step(step_action)

    return env_states


def _generate_queries(
    env,
    candidate_states,
    required_count,
    eval_runs,
    random_open_loop_candidates,
    policy_horizons_candidates,
    policy_a,
    policy_b,
    ignore_delta,
    estimate_db_distance_fn,
    use_wandb=True,
):
    # core attributes
    obss_a = []
    obss_b = []
    actions_a = []
    actions_b = []
    policy_horizons = []
    open_loop_horizons = []
    targets = []

    # info attributes
    returns_a = []
    returns_b = []
    rewards_a = []
    rewards_b = []
    returns_list_a = []
    returns_list_b = []
    policy_horizons_a = []
    policy_horizons_b = []
    policy_actions_a = []
    policy_actions_b = []
    open_loop_horizons_a = []
    open_loop_horizons_b = []
    sim_states_a = []
    sim_states_b = []
    db_distances_a = defaultdict(lambda: [])
    db_distances_b = defaultdict(lambda: [])

    for open_loop_horizon in random_open_loop_candidates:
        for policy_horizon in policy_horizons_candidates:

            current_horizon_count = 0
            decay_count = 0
            distance_max_heap = []
            while current_horizon_count < required_count:
                decay_count += 1
                distance_threshold = max(0.9 - (0.001 * decay_count), 0)
                if len(distance_max_heap) > 0 and (-distance_max_heap[0][0][0]) * 0.2 > distance_threshold:
                    (_, _), query = distance_max_heap.pop()

                    open_loop_horizons.append(query["open_loop_horizon"])
                    policy_horizons.append(query["policy_horizon"])
                    obss_a.append(query["obs_a"])
                    obss_b.append(query["obs_b"])
                    actions_a.append(query["action_a"])
                    actions_b.append(query["action_b"])
                    targets.append(query["target"])

                    returns_a.append(query["info"]["return_a"])
                    returns_b.append(query["info"]["return_b"])
                    rewards_a.append(query["info"]["rewards_a"])
                    rewards_b.append(query["info"]["rewards_b"])
                    returns_list_a.append(query["info"]["rewards_a"])
                    returns_list_b.append(query["info"]["rewards_b"])
                    sim_states_a.append(query["info"]["state_a"])
                    sim_states_b.append(query["info"]["state_b"])

                    open_loop_horizons_a.append(query["info"]["open_loop_horizon_a"])
                    open_loop_horizons_b.append(query["info"]["open_loop_horizon_b"])
                    policy_horizons_a.append(query["info"]["policy_horizon_a"])
                    policy_horizons_b.append(query["info"]["policy_horizon_b"])

                    policy_actions_a.append(query["info"]["policy_actions_a"])
                    policy_actions_b.append(query["info"]["policy_actions_b"])

                    for k in query["info"]["dataset_distance_a"].keys():
                        db_distances_a[k].append(query["info"]["dataset_distance_a"][k])

                    for k in query["info"]["dataset_distance_b"].keys():
                        db_distances_b[k].append(query["info"]["dataset_distance_b"][k])

                    current_horizon_count += 1

                else:
                    same_state = random.choices([True, False], weights=[0.5, 0.5], k=1)[0]
                    # query-a attributes
                    idx = random.randint(0, len(candidate_states["obs"]) - 1)
                    obs_a, sim_state_a = candidate_states["obs"][idx], candidate_states["state"][idx]
                    action_a = [env.action_space.sample() for _ in range(open_loop_horizon)]

                    # query-b attributes
                    if same_state:
                        obs_b = deepcopy(obs_a)
                        sim_state_b = deepcopy(sim_state_a)
                    else:
                        idx = random.randint(0, len(candidate_states["obs"]) - 1)
                        obs_b, sim_state_b = candidate_states["obs"][idx], candidate_states["state"][idx]

                    action_b = [env.action_space.sample() for _ in range(open_loop_horizon)]

                    return_a, _policy_actions_a, _rewards_a, open_loop_horizon_a, policy_horizon_a, db_distance_a = mc_return(
                        env, obs_a, sim_state_a, action_a, policy_horizon, policy_a, eval_runs, estimate_db_distance_fn
                    )

                    return_b, _policy_actions_b, _rewards_b, open_loop_horizon_b, policy_horizon_b, db_distance_b = mc_return(
                        env, obs_b, sim_state_b, action_b, policy_horizon, policy_b, eval_runs, estimate_db_distance_fn
                    )

                    return_a_mean = np.mean(return_a)
                    return_b_mean = np.mean(return_b)
                    open_loop_horizon_a_mean = np.mean(open_loop_horizon_a)
                    open_loop_horizon_b_mean = np.mean(open_loop_horizon_b)
                    policy_horizon_a_mean = np.mean(policy_horizon_a)
                    policy_horizon_b_mean = np.mean(policy_horizon_b)
                    query_distance = np.mean(list(db_distance_a.values()) + list(db_distance_b.values()))
                    value_delta = abs(return_a_mean - return_b_mean)
                    if not (
                        (value_delta <= ignore_delta)
                        or (min(return_b) <= max(return_a) <= max(return_b))
                        or (min(return_b) <= min(return_a) <= max(return_b))
                    ):
                        heapq.heappush(
                            distance_max_heap,
                            (
                                (-query_distance, -value_delta),
                                {
                                    "obs_a": obs_a,
                                    "action_a": action_a[0].tolist(),
                                    "obs_b": obs_b,
                                    "action_b": action_b[0].tolist(),
                                    "open_loop_horizon": open_loop_horizon,
                                    "policy_horizon": policy_horizon,
                                    "target": return_a_mean < return_b_mean,
                                    "info": {
                                        "return_a": return_a_mean,
                                        "return_b": return_b_mean,
                                        "rewards_a": _rewards_a,
                                        "rewards_b": _rewards_b,
                                        "state_a": sim_state_a,
                                        "state_b": sim_state_b,
                                        "runs": eval_runs,
                                        "policy_actions_a": _policy_actions_a,
                                        "policy_actions_b": _policy_actions_b,
                                        "policy_horizon_a": policy_horizon_a_mean,
                                        "policy_horizon_b": policy_horizon_b_mean,
                                        "open_loop_horizon_a": open_loop_horizon_a_mean,
                                        "open_loop_horizon_b": open_loop_horizon_b_mean,
                                        "dataset_distance_a": db_distance_a,
                                        "dataset_distance_b": db_distance_b,
                                        "query_distance": query_distance,
                                    },
                                },
                            ),
                        )
                    if use_wandb:
                        wandb.log({"query-generate-value-delta": value_delta})

    return {
        "obs_a": obss_a,
        "action_a": actions_a,
        "obs_b": obss_b,
        "action_b": actions_b,
        "policy_horizon": policy_horizons,
        "target": targets,
        "info": {
            "return_a": returns_a,
            "return_b": returns_b,
            "rewards_a": rewards_a,
            "rewards_b": rewards_b,
            "state_a": sim_states_a,
            "state_b": sim_states_b,
            "runs": eval_runs,
            "policy_horizon_a": policy_horizons_a,
            "policy_horizon_b": policy_horizons_b,
            "policy_actions_a": policy_actions_a,
            "policy_actions_b": policy_actions_b,
            "dataset_distance_a": dict(db_distances_a),
            "dataset_distance_b": dict(db_distances_b),
        },
    }


def generate_queries(env, candidate_states, policies, args, estimate_db_distance_fn=None):
    _queries = {}
    total_query_count = 0

    policy_ids = sorted(policies.keys())
    for i, policy_id_a in enumerate(policy_ids):
        policy_a = policies[policy_id_a]
        for policy_id_b in args.policy_ids[i + 1 :]:
            policy_b = policies[policy_id_b]
            print(f"policy_a:{policy_a}, policy_b:{policy_b}")
            _key = ((args.env_name, policy_id_a), (args.env_name, policy_id_b))
            _queries[_key] = _generate_queries(
                env,
                candidate_states,
                args.per_policy_comb_query,
                args.eval_runs,
                [1],
                [_ - 1 for _ in args.horizons],
                policy_a,
                policy_b,
                args.ignore_delta,
                estimate_db_distance_fn,
            )

            total_query_count += len(_queries[_key]["obs_a"])

            # log for tracking progress
            if args.use_wandb:
                wandb.log({"query-count": total_query_count})

    return _queries


def main():
    # Let's gather arguments
    parser = argparse.ArgumentParser(description="Generate queries")
    parser.add_argument("--env-name", default="d4rl:maze2d-open-dense-v0", help="name of the environment")
    parser.add_argument("--eval-runs", type=int, default=2, help="monte carlo runs for query evaluation")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="noise to be added to actions during " "exploration for initial query states"
    )
    parser.add_argument(
        "--ignore-delta",
        type=float,
        default=20,
        help="ignore query if difference between two sides" " of query is less than it.",
    )
    parser.add_argument("--horizons", nargs="+", help="horizon lists", type=int, required=True)
    parser.add_argument("--policy-ids", nargs="+", help="policy id lists", type=int, required=True)
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument(
        "--max-trans-count", type=int, default=1000, help="maximum number of transition count for " "initial state candidates"
    )
    parser.add_argument(
        "--save-prob",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--per-policy-comb-query",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    # setup wandb for experiment tracking
    if args.use_wandb:
        wandb.init(project="opcc-diverse", config={"env_name": args.env_name}, save_code=True)

    # seed
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # ################
    # Generate Queries
    # ################
    env = gym.make(args.env_name)
    env.action_space.seed(0)
    env.seed(0)

    policies = {policy_id: opcc.get_policy(args.env_name, policy_id)[0] for policy_id in args.policy_ids}

    # generate candidate states
    candidate_states = generate_candidate_initial_states(
        env, policies, args.max_trans_count, save_prob=0.5, kd_interval=50, action_noise=0.05
    )
    # visualize initial candidates
    pca = PCA(n_components=2, svd_solver="full")
    low_dim_data = pca.fit_transform(candidate_states["obs"])
    fig = px.scatter(x=low_dim_data[:, 0], y=low_dim_data[:, 1])
    _path = os.path.join(ASSETS_DIR, args.env_name, "initial_state_distribution.png")
    fig.write_image(_path)
    if args.use_wandb:
        wandb.log({"candidate-initial-state-pca": fig})

    # create kd-trees for each of the dataset
    kd_dataset = {}
    for dataset_name in opcc.get_dataset_names(args.env_name):
        dataset = opcc.get_qlearning_dataset(args.env_name, dataset_name)
        kd_dataset[dataset_name] = KDTree(np.concatenate((dataset["observations"], dataset["actions"]), 1))

    def estimate_db_distance_fn(obs_action):
        distances = {}
        for dataset_name in kd_dataset:
            kd_tree = kd_dataset[dataset_name]
            distances[dataset_name] = kd_tree.get_knn_batch(obs_action, k=1)
        return distances

    # generate queries
    queries = generate_queries(env, candidate_states, policies, args, estimate_db_distance_fn)

    # visualize data
    returns_a, returns_b, target, horizon = [], [], [], []
    d_names, query_distances = [], []
    for _, v in queries.items():
        returns_a += v["info"]["return_a"]
        returns_b += v["info"]["return_b"]
        target += v["target"]
        horizon += v["policy_horizon"]
        for d_name, distances_a in v["info"]["dataset_distance_a"].items():
            d_names += [d_name for _ in range(len(distances_a))]
            distances_b = v["info"]["dataset_distance_b"][d_name]
            query_distances += [(d_a + d_b) for d_a, d_b in zip(distances_a, distances_b)]
            print(d_names[0], d_names[-1])

    return_fig = px.scatter(
        x=returns_a, y=returns_b, color=target, marginal_x="histogram", marginal_y="histogram", symbol=horizon
    )
    distance_fig = px.histogram(x=query_distances, color=d_names)

    if args.use_wandb:
        wandb.log({"query-values-scatter": return_fig, "distance-histogram": distance_fig})

    # save queries
    _path = os.path.join(ASSETS_DIR, args.env_name, "queries.p")
    os.makedirs(os.path.join(ASSETS_DIR, args.env_name), exist_ok=True)
    pickle.dump(queries, open(_path, "wb"))
    if args.use_wandb:
        wandb.save(_path)
    distance_fig.write_image(os.path.join(ASSETS_DIR, args.env_name,
                                          "distance.png"))
    return_fig.write_image(os.path.join(ASSETS_DIR, args.env_name,
                                        "return.png"))

    # close env
    env.close()


if __name__ == "__main__":
    main()
