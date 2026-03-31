import argparse
from tqdm import tqdm
from autojudger import AutoJuger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--file_path', type=str, required=True)
    return parser.parse_args()

def compute_score(data, metric=None):
    """
    Compute the score of the data using the metric.
    """
    correct_count = 0
    for index, row in tqdm(data.iterrows()):
        if test_equal(row['solution_str'], row['ground_truth']):
            correct_count += 1
    return correct_count / len(data)

def metric_permutation(data, base_metric, target_metric):
    """
    Permute the metric of the data.
    """
    pass