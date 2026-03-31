from base_judger import Judger
import os
import re
import sys
import base64
from collections import defaultdict
import orjson
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json
from datasets import load_dataset
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pickle
import zlib
import pandas as pd
from typing import Union

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.lcb.compute_code_generation_metrics import codegen_metrics, check_correctness
from utils.lcb.pass_k_utils import extract_instance_results
from utils.lcb.code_generation import load_code_generation_dataset

from utils.lcb.testing_utils import run_test

class CodeJudger(Judger):
    def __init__(self, data_source: str = 'lcb'):
        self.data_source = data_source
        if data_source == 'lcb':
            pass
        else:
            pass
    
    def extract_answer(self, model_output: str) -> str:
        r"""Extract the answer from the passage."""
        if model_output.count('```') == 2:
            outputlines = model_output.split("\n")
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
            if len(indexlines) < 2:
                return ""
            return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
        else:
            if "```" not in model_output:
                return model_output
            try:
                pattern = r"```(.*?)\n([\s\S]*?)\n```"
                result = re.findall(pattern, model_output)
                return result[0][1]
            except Exception as e:
                print(f"Error processing output: {e}")
                return model_output

    def _patch_lcb_test_cases(self, sample: dict) -> dict:
        public_test_cases = json.loads(sample['public_test_cases'])
        try:
            private_test_cases = json.loads(sample['private_test_cases'])  # type: ignore
        except:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(sample['private_test_cases'].encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        sample['metadata'] = json.loads(sample['metadata'])
        sample['input_output'] =json.dumps(
                {
                    "inputs": [
                        t['input']
                        for t in public_test_cases + private_test_cases
                    ],
                    "outputs": [
                        t['output']
                        for t in public_test_cases + private_test_cases
                    ],
                    "fn_name": sample['metadata'].get("func_name", None),
                }
            )
        return sample

    def grade_answer(self, given_answer: str, *, ground_truth: str = None, sample: dict = None):
        sample = self._patch_lcb_test_cases(sample)
        debug = False
        timeout = 10  # TODO: move to eval_protocol.yaml

        current_result = [-2]
        try:
            current_result, current_metadata = check_correctness(sample, given_answer, timeout, debug)
            if debug:
                print(f"Successful compilation of task {sample['question_id']}!")
            fixed = []
            for e in current_result:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            current_result = fixed
            if not np.all(current_result):
                if debug:
                    print(f"Results were not True for all test cases {current_result=}\n")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            # break
            current_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(current_result, list), current_result
            assert isinstance(current_metadata, dict), current_metadata

        if debug:
            print("Given Answer\n")
            print(given_answer)
            print("\n")
            print("Result\n")
            print(current_result)
            print("*" * 30 + "\n\n")
        return current_result, current_metadata
    
    def _estimate_pass_at_k(self, num_samples: Union[int, list[int]], num_correct: list[int], k: int):
        def estimator(n: int, c: int, k: int) -> float:
            """Calculates 1 - comb(n - c, k) / comb(n, k)."""
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        import itertools

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array(
            [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
        )

    def _compute_metrics_from_predictions(self, results, k_list=[1, 5]):
        total = []
        correct = []
        idxes = []
        for idx, test_results in results.items():
            all_correct = []
            for test_result in test_results:
                gen = np.array(test_result)
                all_correct.append(np.all(gen > 0))
                
            idxes.append(idx)
            total.append(len(all_correct))
            correct.append(sum(all_correct))
        total = np.array(total)
        correct = np.array(correct)
        ks = k_list
        detail_pass_at_k = {
            f"pass@{k}": self._estimate_pass_at_k(total, correct, k).tolist()
            for k in ks
            if (total >= k).all()
        }
        pass_at_k = {
            f"pass@{k}": self._estimate_pass_at_k(total, correct, k).mean()
            for k in ks
            if (total >= k).all()
        }
        detail_metrics = {k: dict(zip(idxes, v)) for k, v in detail_pass_at_k.items()}
        pass_at_k["detail"] = detail_metrics
        return pass_at_k

    def compute_score(self, data: pd.DataFrame, epochs: int, already_judged=False):
        # excute all tests
        results = defaultdict()
        metadatas = defaultdict()
        inputs = []
        for index, row in tqdm(data.iterrows(), total=len(data), desc='Computing code score'):
            if already_judged and row['judge_result'] is not None:
                results[index], metadatas[index] = row['judge_result']
            else:
                prediction = self.extract_answer(row['model_output'])
                inputs.append((prediction, {"sample": row.to_dict()}, index))
                data.at[index, 'prediction'] = prediction

        if not already_judged or len(results) < len(data):
            with tqdm(total=len(inputs)) as pbar:
                with ProcessPoolExecutor(
                    max_workers=min(64, os.cpu_count())
                ) as executor:
                    futures = {
                        executor.submit(self.grade_answer, model_output, **sample) : index for model_output, sample, index in inputs
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        results[index], metadatas[index] = future.result()
                        pbar.update(1)
            assert len(results) == len(inputs), f"results = {len(results)} inputs = {len(inputs)} {results=}"


        merged_results = defaultdict(list)
        original_data_length = len(data) // epochs
        for idx, result in sorted(results.items(), key=lambda x: x[0]):
            if not already_judged:
                data.at[idx, 'judge_result'] = json.dumps({'result': result, 'metadata': metadatas[idx]})
            merged_results[idx % original_data_length].append(result)
            # final_results.append(result)
            # final_metadatas.append(metadatas[idx])

        # compute metrics
        # TODO: pass@k
        metrics = self._compute_metrics_from_predictions(merged_results)


        return metrics, data


"""
illustation of output_results

{
    "date": "2025-03-14 12:00",
    "pass@1": 0.5,
    "detail_pass@1": {"easy": 0.5, "medium": 0.5, "hard": 0.5},
    "eval": {
        "question_id_1": {
            "code_list": ["code1", "code2", "code3"],
            "graded_list": [True, False, True],
        }
    }
}
"""