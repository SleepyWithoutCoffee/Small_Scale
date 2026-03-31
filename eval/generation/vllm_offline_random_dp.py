from multiprocessing import Process, Queue, set_start_method
import time

from datetime import datetime
import pandas as pd
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import socket
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.utils import offline_eval_formatter, load_config, get_model_path, get_output_path, patch_length, compose_file_name

DEFAULT_ENGINE_ARGS = {"gpu_memory_utilization": 0.95, "tensor_parallel_size": 1}
DEFAULT_SAMPLING_PARAMS = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096, "seed": None, "skip_special_tokens": False, "spaces_between_special_tokens": False}

# TODO: override the apply_chat_template function in tokenizer to keep thinking content
# NOTE: the original apply_chat_template will omit thinking content from assistent history messages, which works for single turn but is wrong for multi-turn
# def apply_chat_template(prompts: list[list[dict[str, str]]], tokenizer: AutoTokenizer, enable_thinking: bool = True) -> list[str]:

import numpy as np

def start(rank: int, floor: int, remainder: int) -> int:
    return rank * floor + min(rank, remainder)

def calculate_interval(rank: int, dp_size: int, total_len: int) -> tuple[int, int]:
    floor = total_len // dp_size
    remainder = total_len % dp_size
    return start(rank, floor, remainder), start(rank + 1, floor, remainder)

def worker_main(
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    result_queue: Queue,
    eval_data: pd.DataFrame,
    model_path: str,
    engine_args: dict,
    sampling_params: dict,
    dp_size: int,
    random_sample: bool = False
) -> None:
    """
    Worker process for distributed inference.
    Returns results through the result_queue.
    """
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Get the data slice for this worker
    start_idx, end_idx = calculate_interval(global_dp_rank, dp_size, len(eval_data))
    rank_prompts = eval_data['prompt'].tolist()[start_idx:end_idx]

    if len(rank_prompts) == 0:
        rank_prompts = [[{"role": "user", "content": ""}]]

    sampling_params = SamplingParams(**sampling_params)

    try:
        if random_sample:
            engine_seed = int(time.time() * 1000000)
            engine_args['seed'] = engine_seed
        llm = LLM(model=model_path,
                tokenizer=model_path,
                trust_remote_code=True,
                **engine_args)

        outputs = llm.chat(
            messages=rank_prompts,
            sampling_params=sampling_params,
            use_tqdm=True
        )

        # Extract the generated text from outputs
        generated_texts = []
        for output in outputs:
            generated_texts.append(output.outputs[0].text)

        # Send results back to parent process
        result_queue.put({
            'rank': global_dp_rank,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'outputs': generated_texts,
            'success': True
        })

    except Exception as e:
        # Send error information back to parent process
        result_queue.put({
            'rank': global_dp_rank,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'outputs': [],
            'success': False,
            'error': str(e)
        })
    
    time.sleep(1)


def collect_and_merge_results(result_queue: Queue, dp_size: int, eval_data: pd.DataFrame, original_indices: np.ndarray = None) -> pd.DataFrame:
    """
    Collect results from all worker processes and merge them back into the original dataframe.
    If original_indices is given, restore the original order after merging results.
    """
    # Initialize result storage
    all_results = {}
    successful_ranks = set()
    failed_ranks = set()

    # Collect results from all processes
    for _ in range(dp_size):
        try:
            result = result_queue.get(timeout=None)
            rank = result['rank']

            if result['success']:
                all_results[rank] = result['outputs']
                successful_ranks.add(rank)
                print(
                    f"Rank {rank}: Successfully processed {len(result['outputs'])} samples")
            else:
                failed_ranks.add(rank)
                print(
                    f"Rank {rank}: Failed with error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error collecting result: {e}")
    
    # Merge results back into the dataframe
    merged_data = eval_data.copy()
    merged_data['model_output'] = [''] * len(merged_data)

    cnt = 0
    for rank in successful_ranks:
        start_idx, end_idx = calculate_interval(rank, dp_size, len(eval_data))
        for i, result in enumerate(all_results[rank]):
            if start_idx + i < len(merged_data):
                merged_data.iloc[start_idx + i, merged_data.columns.get_loc('model_output')] = result
                cnt += 1
    if cnt != len(merged_data):
        print(f"Warning: Successfully processed {cnt} samples, but expected {len(merged_data)} samples")
        return None

    print(f"Successfully processed {len(successful_ranks)}/{dp_size} ranks")
    # If shuffling happened, restore original order
    if original_indices is not None:
        merged_data = merged_data.copy()
        merged_data['__original_idx__'] = original_indices
        merged_data = merged_data.sort_values('__original_idx__').drop(columns=['__original_idx__']).reset_index(drop=True)
    return merged_data


def get_open_port() -> int:
    port = os.environ.get("VLLM_PORT")

    if port is not None:
        port = int(port)
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1
                print(f"Port {port-1} is already in use, trying port {port}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def parse_args():
    parser = argparse.ArgumentParser(description="generation_code")

    parser.add_argument('--config', type=str, required=True,
                        help='Path of the configuration file.')
    parser.add_argument('--dataset_name', type=str, required=False,
                        help='Path of the data source file. Override the data source in the config file.')
    parser.add_argument('--model_name', type=str, required=False,
                        help='Name of the model. Override the model name in the config file.')
    # parser.add_argument('--dev', type=bool, required=False,
    #                     help='Whether the model is in development mode. Default is False.')
    return parser.parse_args()


def run(args):
    set_start_method('spawn', force=True)
    config = load_config(args.config)

    dataset_name = args.dataset_name or config.get('dataset_name', None)
    model_name = args.model_name or config.get('model_name', None)
    if dataset_name is None:
        raise ValueError("Dataset name is required.")
    if model_name is None:
        raise ValueError("Model name is required.")

    engine_args = config.get('engine_args', DEFAULT_ENGINE_ARGS)
    sampling_params = config.get('sampling_params', DEFAULT_SAMPLING_PARAMS)

    dp_size = config.get('dp_size', 1)  # TODO: unify dp_size and data_parallel_size
    dp_shuffle = config.get('dp_shuffle', True)

    num_epochs = config.get('num_epochs', 1)

    generation_log_dir = config.get('generation_log_dir', None)

    time_stamp = datetime.now().strftime("%m%d%H%M")
    eval_data, task = offline_eval_formatter(dataset_name)
    log_file_path = compose_file_name(
        generation_log_dir, task, dataset_name, model_name, f'{time_stamp}_{num_epochs}.log')
    with open(log_file_path, "w") as log_file:
        log_file.write(f'{json.dumps(config, indent=4)}\n\n')

    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if num_epochs > 1:
        eval_data = pd.concat([eval_data] * num_epochs, ignore_index=True)

    # ---------- SHUFFLE DATA BEFORE DISTRIBUTION ----------
    if dp_shuffle:
        # np.random.seed(42)
        shuffled_indices = np.random.permutation(eval_data.index.values)
        eval_data = eval_data.iloc[shuffled_indices].reset_index(drop=True)
        original_indices = shuffled_indices
    else:
        original_indices = eval_data.index.values

    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    # Create a queue for collecting results
    result_queue = Queue()

    # Start worker processes
    procs = []
    for local_dp_rank, global_dp_rank in enumerate(range(dp_size)):
        p = Process(
            target=worker_main,
            args=(
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                result_queue,
                eval_data,
                model_path,
                engine_args,
                sampling_params,
                dp_size
            )
        )
        procs.append(p)
        p.start()
        print(f"Started process for rank {global_dp_rank}")

    print("All processes launched. Waiting for results...")

    # Collect and merge results
    final_results = collect_and_merge_results(result_queue, dp_size, eval_data, original_indices=original_indices)

    # Wait for all processes to complete
    for p in procs:
        p.join()
    for p in procs:
        if p.is_alive():
            p.kill()

    if final_results is None:
        print("Generation failed. Exiting...")
        exit(1)

    # Save results
    _, data_output_path = get_output_path()
    output_file_path = compose_file_name(
        data_output_path, task, dataset_name, model_name, f'{time_stamp}_{num_epochs}.parquet')
    print("Adding tokenized outputs...")
    final_results = patch_length(final_results, tokenizer)
    final_results.to_parquet(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")

    print("Generation completed successfully!")
    return final_results, output_file_path


if __name__ == "__main__":
    args = parse_args()
    _, output_file_path = run(args)
    with open("/path/to/eval/temp/gen_id.txt", "w") as f:
        f.write(output_file_path)