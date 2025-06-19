import os
import json
import yaml
import multiprocessing

from openai import OpenAI
from box import Box
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils import CHAT_MODELS, partition


client = None


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def query_loop(model, instance, temperature, max_new_tokens):
    # Edge case: If an empty string is passed, return an empty string
    if instance == "":
        return ""

    if any(chat_model_name in model for chat_model_name in CHAT_MODELS):
        response = (
            client.chat.completions.create(
                messages=[{"role": "user", "content": instance}],
                model=model,
                # temperature=temperature,
                max_completion_tokens=max_new_tokens,
                # max_tokens=max_new_tokens,
            )
            .choices[0]
            .message.content
        )
    else:
        response = (
            client.completions.create(
                model=model, prompt=instance, temperature=temperature
            )
            .choices[0]
            .text
        )
        
    return response

def query_worker(model, inputs, process_id, save_dir, lock, temperature, max_new_tokens):
    with lock:
        bar = tqdm(
            desc=f"Process {process_id+1}",
            total=len(inputs),
            position=process_id + 1,
            leave=False,
        )

    # If partially populated results file exists, load and continue
    responses = json.load(open(save_dir, "r")) if os.path.exists(save_dir) else list()
    start_index = len(responses)

    for instance in inputs[start_index:]:
        with lock:
            bar.update(1)

        # If instance is a list, pass contents one by one. Otherwise, pass instance
        if type(instance) == list:
            response = [
                (
                    query_loop(model, instance_item, temperature, max_new_tokens)
                    if instance_item != ""
                    else ""
                )
                for instance_item in instance
            ]
        elif type(instance) == str:
            response = query_loop(model, instance, temperature, max_new_tokens)

        responses.append(response)

        json.dump(responses, open(save_dir, "w"), indent=4)

    with lock:
        bar.close()

    return responses


class OpenAIModel:
    def __init__(self, model_name):
        global client
        config = Box(yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r"))) 
        config.update(
            Box(
                yaml.safe_load(
                    open(os.path.join(os.path.dirname(__file__), "../config.yaml"), "r")
                )
            )
        )        
        keys = json.load(open("keys.json"))

        client = OpenAI(api_key=keys["openai_key"])
        
        self.model_name = model_name.split(":")[-1]
        
        self.num_processes = config.num_processes
        self.max_new_tokens = config.max_new_tokens
        
    def infer(self, instance, temperature=0):
        return query_loop(self.model_name, instance, temperature, self.max_new_tokens)

    def infer_batch(self, inputs, save_dir, temperature=0):
        # Partition instances
        paritioned_inputs = partition(inputs, self.num_processes)

        # Start multiprocess instances
        lock = multiprocessing.Manager().Lock()
        pool = multiprocessing.Pool(processes=self.num_processes)

        worker_results = []
        for process_id in range(len(paritioned_inputs)):
            async_args = (
                self.model_name,
                paritioned_inputs[process_id],
                process_id,
                save_dir.replace(
                    "." + save_dir.split(".")[-1],
                    f"-process={process_id}{'.' + save_dir.split('.')[-1]}",
                ),
                lock,
                temperature,
                self.max_new_tokens,
            )

            # Run each worker
            worker_results.append(pool.apply_async(query_worker, args=async_args))

        pool.close()
        pool.join()

        responses = []
        for worker_result in worker_results:
            responses += worker_result.get()

        return responses
