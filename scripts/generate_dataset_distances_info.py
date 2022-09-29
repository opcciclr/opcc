
import shutil
import os
import argparse
import pickle

import numpy as np
from kd import KDTree



if __name__ == '__main__':

    # Let's gather arguments
    parser = argparse.ArgumentParser(
        description="Calculate query distances from dataset")
    parser.add_argument("--env-name",
                        default="d4rl:maze2d-open-dense-v0",
                        help="name of the environment")

    args = parser.parse_args()

    import opcc

    queries = opcc.get_queries(args.env_name)

    # create kd-trees for each of the dataset
    for dataset_name in opcc.get_dataset_names(args.env_name):
        dataset = opcc.get_qlearning_dataset(args.env_name, dataset_name)
        kd_dataset = KDTree(np.concatenate((dataset["observations"],
                                            dataset["actions"]), 1))
        distance_info = {}
        for query_key, query_batch in queries.items():

            distance_info[query_key] = kd_dataset.get_knn_batch(
                np.concatenate((query_batch['obs_a'], query_batch['action_a']), 1),
                                                                k=1)
        dataset_dir_path = os.path.join(opcc.ASSETS_DIR,
                                        args.env_name, 'dataset_distances',
                                        dataset_name)
        if os.path.exists(dataset_dir_path):
            shutil.rmtree(dataset_dir_path)

        os.makedirs(dataset_dir_path)
        distances_path = os.path.join(dataset_dir_path, 'distances.p')
        distances_split_dir = os.path.abspath(os.path.join(dataset_dir_path, 'distances'))
        pickle.dump(distance_info, open(distances_path, 'wb'))
        os.makedirs(distances_split_dir)

        query_checksum_path = os.path.abspath(os.path.join(opcc.ASSETS_DIR,
                                           args.env_name,
                                           'queries','md5sums.txt'))
        os.system(f" split -b25M -e {distances_path.abs()}  "
                  f"{distances_split_dir}/distances.")
        os.system(f" cp  {query_checksum_path} > "
                  f"{distances_split_dir}/queries_md5sums.txt")
        os.system(f" md5sum {distances_path.abs()} > "
                  f"{distances_split_dir}/distances_md5sums.txt")