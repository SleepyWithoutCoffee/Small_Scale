import sys
import os
sys.path.append(os.path.dirname(__file__))
from api_utils import openai_utils

class Router():
    def __init__(self, llm_as_judge):
        # self.epochs = llm_as_judge["epochs"]
        self.provider = llm_as_judge['judge_model']['api_type']
        self.api_key = llm_as_judge['judge_model']['api_key']
        self.api_url = llm_as_judge['judge_model']['api_url']
        self.route()

    def route(self):
        """
        send_one_request: model, messages, error_query_save_path, (**)model_kwargs
            including - messages already formatted; retry logic
        concurrent_send_request: model, queries, prompt_template, error_query_save_path, system_prompt, image_paths, histories=None, max_workers, batch_size, output_path, (**)model_kwargs
        """
        if self.provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key
            os.environ["OPENAI_URL"] = self.api_url
            self.send_one_request = openai_utils.send_one_request_to_openai
            self.concurrent_send_requests = openai_utils.concurrent_send_requests