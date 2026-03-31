import json
import pandas
from tqdm import tqdm
import os
import requests
from jinja2 import Template
import concurrent.futures
import re
from time import sleep
import base64

default_system_prompt = ""
default_prompt_template = "User Query:\n{{query}}\n\nPlease think step by step to analyze the intent, and put your final answer (eg, 1, 2, 3) within \\boxed{}."

TEMP_BATCH_DIR = "/path/to/webdev/MyJudge/utils/temp"

def send_one_request_to_openai(model, messages, error_query_save_path, max_tokens=8192):
    initial_sleep_time = 5
    initial_timeout_time = 240

    sleep_time = initial_sleep_time
    timeout_time = initial_timeout_time

    wait_cnt = 0
    none_cnt = 0
    timeout_cnt = 0

    error_503_limit = 2
    error_429_limit = 5
    none_limit = 5
    timeout_limit = 5
    wait_limit = 15

    temperature = 0.6

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }
    json_data = {"model": model,
                 "messages": messages,
                 "temperature": temperature,
                 "max_tokens": max_tokens,
                 }
    while True:
        if none_cnt != 0:
            print("Recieved NONE response...")
        if none_cnt >= none_limit or timeout_cnt >= timeout_limit:
            break
        try:
            raw_response = requests.post(
                os.environ['OPENAI_URL'],
                headers=headers,
                json=json_data,
                timeout=timeout_time
            )
        except:
            timeout_cnt += 1
            print(f"Time out...sleep for {sleep_time}s")
            sleep(sleep_time)
            timeout_time += 120
            continue

        if raw_response is None:
            none_cnt += 1
            continue

        if raw_response.status_code == 200:
            wait_cnt = 0
            timeout_cnt = 0

            sleep_time = initial_sleep_time
            timeout_time = initial_timeout_time

            try:
                response = json.loads(raw_response.text)
                response_content = response['choices'][0]['message']['content']
            except:
                none_cnt += 1
                continue

            if response_content.strip() != "":
                none_cnt = 0
                break
            else:
                none_cnt += 1
        else:
            print(f"Error: {raw_response.status_code}")
            print(raw_response.headers)

            print(f'Wait for {sleep_time} s...')
            if (wait_cnt >= error_503_limit and raw_response.status_code == 503) or wait_cnt >= wait_limit:
                if wait_cnt >= wait_limit:
                    print(f'Max retry limit reached, exit')
                else:
                    print(f'Poison Pill, exit')
                break
            elif wait_cnt >= error_429_limit and raw_response.status_code == 429:
                sleep_time += 10
                wait_cnt += 1
                sleep(sleep_time)
            else:
                wait_cnt += 1

    if wait_cnt != 0 or none_cnt != 0 or timeout_cnt != 0:
        error_list = []
        if wait_cnt >= error_503_limit and wait_cnt < wait_limit:
            error_list.append("Poison Pill")
        elif wait_cnt >= wait_limit:
            error_list.append("Max Retry")
        
        if none_cnt >= none_limit:
            error_list.append("None Response")

        if timeout_cnt >= timeout_limit:
            error_list.append("Timeout")

        if error_query_save_path is not None:
            save_message = ""
            for message in messages:
                if message["role"] == "user":
                    if isinstance(message["content"], list):
                        save_message = message["content"][0]['text']
                    else:
                        save_message = message["content"]
                    break
            with open(error_query_save_path, "a") as f:
                json_str = json.dumps({"error": error_list, "query": save_message}, ensure_ascii=False) + "\n"
                f.write(json_str)
        raise Exception("Poison Pill Exception")

    return response_content

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_message(query, prompt_template=default_prompt_template, system_prompt=None, image_path=None, history=None):
    template = Template(prompt_template)
    content = template.render(query=query)
    mllm = image_path is not None
    if mllm:
        try:
            base64_image = encode_image(image_path)
        except:
            print(f"image_path does not exist: f{image_path}")
            mllm = False
    if not mllm:
        return_message = [
                {
                    "role": "user",
                    "content": content
            }]
    else:
        return_message = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "detail": "auto",
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
                ]
            }
        ]
    if system_prompt is not None:
        return_message.insert(0, {"role": "system", "content": system_prompt})
    if history is not None:
        return_message = history + return_message
    
    return return_message

def concurrent_send_requests(model, queries, prompt_template, error_query_save_path=None, system_prompt=None, image_paths=None, histories=None, max_workers=32, batch_size=None, output_path=None, max_tokens=16384):
    results = {}

    total_num = len(queries)
    if not isinstance(system_prompt, list):
        system_prompt = [system_prompt] * total_num
    if image_paths is None:
        image_paths = [None] * total_num
    if histories is None:
        histories = [None] * total_num

    extra_param_list = []
    for i in range(total_num):
        extra_param_list.append({
                "system_prompt": system_prompt[i],
                "image_path": image_paths[i],
                "history": histories[i]
            }
        )

    if batch_size is None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(send_one_request_to_openai, model, format_message(query, prompt_template, **extra_param), error_query_save_path, max_tokens): query for query, extra_param in zip(queries, extra_param_list)}
            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item)):
                try:
                    query = future_to_item[future]
                    result = future.result()
                    results[query] = result
                except Exception as e:
                    print(f"Error processing item: {e}")
    else:
        total_batches = (total_num + batch_size - 1) // batch_size

        start_batch_idx = 0
        if TEMP_BATCH_DIR and os.path.exists(TEMP_BATCH_DIR):
            existing_files = [f for f in os.listdir(TEMP_BATCH_DIR) if f.startswith("batch_") and f.endswith(".json")]
            if existing_files:
                for f in existing_files:
                    batch_file_path = os.path.join(TEMP_BATCH_DIR, f)
                    with open(batch_file_path, "r", encoding="utf-8") as fp:
                        batch_results = json.load(fp)
                        results.update(batch_results)
                completed_batch_indices = [int(f.replace("batch_", "").replace(".json", "")) for f in existing_files]
                start_batch_idx = max(completed_batch_indices) + 1
                print(f"Found {len(existing_files)} completed batches, resuming from batch {start_batch_idx + 1}")

        for batch_idx in range(start_batch_idx, total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_num)
            batch_queries = queries[batch_start:batch_end]
            batch_extra_params = extra_param_list[batch_start:batch_end]
            batch_results = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {executor.submit(send_one_request_to_openai, model, format_message(query, prompt_template, **extra_param), error_query_save_path): query for query, extra_param in zip(batch_queries, batch_extra_params)}
                for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item), desc=f"Batch {batch_idx + 1}/{total_batches}"):
                    try:
                        query = future_to_item[future]
                        result = future.result()
                        batch_results[query] = result
                        results[query] = result
                    except Exception as e:
                        print(f"Error processing item: {e}")

            if TEMP_BATCH_DIR:
                os.makedirs(TEMP_BATCH_DIR, exist_ok=True)
                batch_file_path = os.path.join(TEMP_BATCH_DIR, f"batch_{batch_idx}.json")
                with open(batch_file_path, "w", encoding="utf-8") as fp:
                    json.dump(batch_results, fp, ensure_ascii=False, indent=4)
                print(f"Batch {batch_idx + 1} saved to {batch_file_path}")

        if TEMP_BATCH_DIR and os.path.exists(TEMP_BATCH_DIR):
            if output_path:
                with open(output_path, "w", encoding="utf-8") as fp:
                    json.dump(results, fp, ensure_ascii=False, indent=4)
                print(f"All results saved to {output_path}")

            for f in os.listdir(TEMP_BATCH_DIR):
                if f.startswith("batch_") and f.endswith(".json"):
                    os.remove(os.path.join(TEMP_BATCH_DIR, f))
            print(f"Temporary batch files cleaned up")

    return results

def serial_send_requests(model, queries, prompt_template, error_query_save_path=None, system_prompt=None, image_paths=None, histories=None, batch_size=None):
    results = {}

    total_num = len(queries)
    if not isinstance(system_prompt, list):
        system_prompt = [system_prompt] * total_num
    if image_paths is None:
        image_paths = [None] * total_num
    if histories is None:
        histories = [None] * total_num

    extra_param_list = []
    for i in range(total_num):
        extra_param_list.append({
                "system_prompt": system_prompt[i],
                "image_path": image_paths[i],
                "history": histories[i]
            }
        )

    if batch_size is None:
        batch_size = total_num

    for batch_start in range(0, total_num, batch_size):
        batch_end = min(batch_start + batch_size, total_num)
        batch_queries = queries[batch_start:batch_end]
        batch_extra_params = extra_param_list[batch_start:batch_end]

        for query, extra_param in tqdm(zip(batch_queries, batch_extra_params), total=len(batch_queries), desc=f"Batch {batch_start//batch_size + 1}"):
            try:
                result = send_one_request_to_openai(model, format_message(query, prompt_template, **extra_param), error_query_save_path)
                results[query] = result
            except Exception as e:
                print(f"Error processing item: {e}")

    return results
