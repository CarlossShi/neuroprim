import argparse
import numpy as np
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--graph_size', type=int, default=20)
    parser.add_argument('--max_dataset_size', type=int, default=10000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    seed = args.seed
    graph_size = args.graph_size
    max_dataset_size = args.max_dataset_size
    np.random.seed(seed)
    data = np.random.uniform(size=(max_dataset_size, graph_size, 2))
    os.makedirs('data/random', exist_ok=True)
    with open('data/random/{}_test_seed{}.pkl'.format(graph_size, seed), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
