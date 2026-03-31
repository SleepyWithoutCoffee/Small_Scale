import time
from datetime import datetime
import pandas as pd
import argparse
import sglang as sgl
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.utils import offline_eval_formatter, load_config, get_model_path, get_output_path, patch_length, compose_file_name

DEFAULT_ENGINE_ARGS = {"tp_size": 1, "dp_size": 1, "random_seed": None}
DEFAULT_SAMPLING_PARAMS = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 4096, "skip_special_tokens": False, "spaces_between_special_tokens": False}


def generate(
    eval_data: pd.DataFrame,
    model_path: str,
    engine_args: dict,
    sampling_params: dict,
    chat_template_args: dict | None = None
) -> None:
    """
    Worker process for distributed inference.
    Returns results through the result_queue.
    """


    llm = sgl.Engine(model_path=model_path,
                tokenizer_path=model_path,
                trust_remote_code=True,
                **engine_args)
    
    if chat_template_args is not None:
        text = llm.tokenizer_manager.tokenizer.apply_chat_template(eval_data['prompt'].tolist(), tokenize=False, add_generation_prompt=True, **chat_template_args)
    else:
        text = llm.tokenizer_manager.tokenizer.apply_chat_template(eval_data['prompt'].tolist(), tokenize=False, add_generation_prompt=True)
    
    start_time = time.time()
    outputs = llm.generate(
        text,
        sampling_params
    )
    end_time = time.time()
    interval = int(end_time - start_time)
    hours = interval // 3600
    minutes = (interval % 3600) // 60
    seconds = interval % 60
    print(f"Generation time: {hours}:{minutes:02d}:{seconds:02d}")

    # Extract the generated text from outputs
    model_outputs = []
    for output in outputs:
        model_outputs.append(output['text'])
    return model_outputs

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
    # set_start_method('spawn', force=True)
    config = load_config(args.config)

    dataset_name = args.dataset_name or config.get('dataset_name', None)
    model_name = args.model_name or config.get('model_name', None)
    if dataset_name is None:
        raise ValueError("Dataset name is required.")
    if model_name is None:
        raise ValueError("Model name is required.")

    engine_args = config.get('engine_args', DEFAULT_ENGINE_ARGS)
    sampling_params = config.get('sampling_params', DEFAULT_SAMPLING_PARAMS)
    chat_template_args = config.get('chat_template_args', None)

    num_epochs = config.get('num_epochs', 1)

    generation_log_dir = config.get('generation_log_dir', None)

    time_stamp = datetime.now().strftime("%m%d%H%M")
    eval_data, task = offline_eval_formatter(dataset_name)
    log_file_path = compose_file_name(
        generation_log_dir, task, dataset_name, model_name, f'{time_stamp}_{num_epochs}.log')
    with open(log_file_path, "w") as log_file:
        log_file.write(f'{config}\n\n')

    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if num_epochs > 1:
        eval_data = pd.concat([eval_data] * num_epochs, ignore_index=True)

    model_outputs = generate(
        eval_data,
        model_path,
        engine_args,
        sampling_params,
        chat_template_args
    )

    if model_outputs is None:
        print("Generation failed. Exiting...")
        exit(1)

    eval_data['model_output'] = model_outputs

    # Save results
    _, data_output_path = get_output_path()
    output_file_path = compose_file_name(
        data_output_path, task, dataset_name, model_name, f'{time_stamp}_{num_epochs}.parquet')
    print("Adding tokenized outputs...")
    eval_data = patch_length(eval_data, tokenizer)
    eval_data.to_parquet(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")
    return eval_data, output_file_path


if __name__ == "__main__":
    args = parse_args()
    _, file_path = run(args)
    with open('/path/to/eval/temp/gen_id.txt', 'w') as f:
        f.write(file_path)