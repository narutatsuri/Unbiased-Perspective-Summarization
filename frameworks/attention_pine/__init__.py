import os
import json
import yaml

import torch
from box import Box
from tqdm import tqdm
from huggingface_hub import login
from transformers import LlamaConfig
from pine import LlamaForCausalLMWithPS, LlamaTokenizerFastWithPS


class PINE:
    def __init__(self, config_path):
        login(json.load(open("keys.json"))["huggingface_token"])
        self.config = Box(yaml.safe_load(open(config_path, "r")))
                
        # Initialize tokenizer and model
        backbone = self.config.backbone.model.split(":")[-1]
        
        model_config = LlamaConfig.from_pretrained(backbone)
        model_config._attn_implementation = 'eager'
        self.tokenizer = LlamaTokenizerFastWithPS.from_pretrained(backbone)
        self.model = LlamaForCausalLMWithPS.from_pretrained(backbone, torch_dtype=torch.float16, config=model_config, device_map="auto")
        
        self.model.generation_config.max_new_tokens = self.config.max_new_tokens
        self.model.generation_config.do_sample = False # Suppose we sue Greedy Decoding
        self.model.generation_config.pad_token_id = 128004 # For llama 3
        
    def infer_batch(self, data, save_dir):
        """
        Run inference on a batch of data. Generate text for each stance. Save results.

        This method iterates over data instances. For each instance, it generates text for 
        each stance using a model. Results are saved as a JSON file at `save_dir`.

        Parameters:
        -----------
        data : list
            List of dictionaries. Each dictionary represents an instance with source text 
            for different stances (e.g., "left_source", "right_source").

        save_dir : str
            Path to save the resulting JSON file. If the directory does not exist, it is created.

        Returns:
        --------
        None
            Saves generated outputs to `save_dir` in JSON format. Adds generated text for each 
            stance to each instance.

        Raises:
        -------
        OSError
            Raised if there is an issue creating the directory or writing to the file.
        """        
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        for instance in tqdm(data, ncols=80, leave=False, desc="PINE"):
            # Format according to PINE documentation
            input_string = [
                self.config.instruction.format(instance["stance"], instance["stance"].capitalize()),
                instance[f"input"].split("\n"),
                "<eos>"
            ]

            inputs = {key: torch.tensor(val).to("cuda") for key, val in self.tokenizer(input_string).items()}
            outputs = self.model.generate(**inputs)                                
            instance[f"generated"] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("<eos>", 1)[-1].split(f"The {instance["stance"].capitalize()}", 1)[-1]

            
            json.dump(data, open(save_dir, "w"), indent=4)
