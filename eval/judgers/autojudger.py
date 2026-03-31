from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import pandas as pd
import sys
import os
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.utils import get_dataset_info, load_config, get_model_path

METRICS = ['acc', 'avg@k']


class AutoJuger:
    def __init__(self, config_path, file_path=None, online_info=None, offline=True):
        """
        online_info: {
            "dataset_name": str,
            "model_name": str,
            "time_stamp": str,
            "data": pd.DataFrame,
            "num_epochs": int,
            "tokenizer": AutoTokenizer,
        }
        """
        config = load_config(config_path)
        max_tokens = config.get('max_tokens', 32768)
        metric = config.get('metric', None)
        prune = config.get('prune', False)
        result_log_dir = config.get('result_log_dir', None)
        self.llm_as_judge = config.get('llm_as_judge', None)
        if offline:
            if file_path is None:
                raise ValueError("file_path is required when offline is True")
            if bool(re.match(r'^\d{8}_\d+\.parquet$', file_path.split('/')[-1])):
                # task = file_path.split('/')[-4]
                dataset_name = file_path.split('/')[-3]
                model_name = file_path.split('/')[-2]
                num_epochs = file_path.split(
                    '/')[-1].split('.')[0].split('_')[-1]
                time_stamp = file_path.split(
                    '/')[-1].split('.')[0].split('_')[0]
            else:  # Compatibility with the naming rules of the old testing framework
                file_name = file_path.split('/')[-1]
                dataset_name = file_name.split('_')[0]
                model_name = file_name.split('_')[1]
                num_epochs = file_name.split('_')[2]
                file_name = file_name.split('.')[0].split('_')[-2]
                time_stamp = 00000000
            data = pd.read_parquet(file_path)
            model_path = get_model_path(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        if not offline:
            if online_info is None:
                raise ValueError(
                    "online_info is required when offline is False")
            dataset_name = online_info['dataset_name']
            model_name = online_info['model_name']
            time_stamp = online_info['time_stamp']
            data = online_info['data']
            num_epochs = online_info['num_epochs']
            tokenizer = online_info['tokenizer']

        _, default_metric, _, task, _ = get_dataset_info(dataset_name, 'test')

        judger = self._get_offline_judger(task)
        if prune:
            for index, row in tqdm(data.iterrows(), total=len(data), desc='Pruning data'):
                if max_tokens < len(row['tokenized_model_output']):
                    pruned_tokenized_model_output = row['tokenized_model_output'][:max_tokens]
                    pruned_model_output = tokenizer.decode(
                        pruned_tokenized_model_output)
                    data.at[index, 'tokenized_model_output'] = pruned_tokenized_model_output
                    data.at[index, 'model_output'] = pruned_model_output

        self.offline = offline
        self.file_path = file_path
        self.data = data
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.time_stamp = time_stamp
        if metric is None:
            self.metric = default_metric
        else:
            self.metric = metric
        self.task = task
        self.judger = judger

        self.max_tokens = int(max_tokens)
        self.num_epochs = int(num_epochs)

        self.result_log_dir = result_log_dir

    def _get_offline_judger(self, task):
        if task == 'math':
            from math_judger import MathJudger
            return MathJudger(self.llm_as_judge)
        elif task == 'mcq':
            from mcq_judger import McqJudger
            return McqJudger()
        elif task == 'code':
            from code_judger import CodeJudger
            return CodeJudger()
        else:
            raise ValueError(f"Unsupported task: {task}")

    def compute_score(self, already_judged=False):  # default compute_score
        """
        Compute the score of the data.
        """
        correct_count = 0
        for index, row in tqdm(self.data.iterrows(), total=len(self.data), desc='Computing score'):
            if already_judged and row['judge_result'] is not None:
                judge_result = row['judge_result']
            else:
                prediction = self.judger.extract_answer(row['model_output'])
                self.data.at[index, 'prediction'] = prediction
                judge_result = self.judger.grade_answer(
                    prediction, ground_truth=str(row['ground_truth']), sample=row)
                self.data.at[index, 'judge_result'] = judge_result
            if judge_result:
                correct_count += 1
        return correct_count / len(self.data)

    def compute_length(self):
        """
        Compute the length of the data.
        """
        max_length = 0
        min_length = self.max_tokens
        average_length = 0
        for index, row in tqdm(self.data.iterrows(), total=len(self.data), desc='Computing length'):
            length = len(row['tokenized_model_output'])
            max_length = max(max_length, length)
            min_length = min(min_length, length)
            average_length += length
        average_length /= len(self.data)
        return average_length, max_length, min_length

    def run(self, already_judged=False):
        # TODO: directly compute with predictions or judge_result in the data
        average_length, max_length, min_length = self.compute_length()
        if self.task != 'code':
            score = self.compute_score()
        else:
            metrics, self.data = self.judger.compute_score(
                # override compute_score with judger's
                self.data, self.num_epochs, already_judged)
            score = metrics['pass@1']  # TODO: pass@k
            backup_log_file_path = os.path.join(
                self.result_log_dir, 'backup', f'{self.dataset_name}.log')

            with open(backup_log_file_path, 'a') as f:
                f.write(f'{self.model_name} | score: {score * 100:.2f} | avg.len: {average_length:.2f} | max.len: {max_length} | min.len: {min_length} | time_stamp: {self.time_stamp} | num_epochs: {self.num_epochs}\n')
                f.write(json.dumps(metrics, ensure_ascii=False, indent=4))
                f.write('\n\n')
        results_log_file_path = os.path.join(
            self.result_log_dir, f'{self.dataset_name}.log')

        with open(results_log_file_path, 'a') as f:
            f.write(f'{self.model_name} | score: {score * 100:.2f} | avg.len: {average_length:.2f} | max.len: {max_length} | min.len: {min_length} | time_stamp: {self.time_stamp} | num_epochs: {self.num_epochs}\n')
        print(f'{self.model_name} | score: {score * 100:.2f} | avg.len: {average_length:.2f} | max.len: {max_length} | min.len: {min_length} | time_stamp: {self.time_stamp} | num_epochs: {self.num_epochs}')

        if self.offline:
            self.data.to_parquet(self.file_path)
        return self.data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--file_path', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    autojudger = AutoJuger(args.config, args.file_path)
    autojudger.run()
