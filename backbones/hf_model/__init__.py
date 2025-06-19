import os
import json
import yaml

import torch
from box import Box
from tqdm import tqdm
from huggingface_hub import login
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import raise_error, GENERATION_MODELS
from .augments import SteeringHiddenState


class HFModel:
    def __init__(self, model_name, config=None):
        
        if not torch.cuda.is_available():
            raise_error("No GPUs available for use")
            
        self.model_name = model_name.split(":")[-1]
        (self.config, empty_config) = (config, False) if config is not None else (Box(), True)
        self.config.update(Box(yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "../config.yaml"), "r"))))      
     
        keys = json.load(open("keys.json"))
        login(keys["huggingface_token"])

        self.device = torch.device("cuda")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.open_ended_generation = any([family in self.model_name.lower() for family in GENERATION_MODELS])
        
        # TODO: Figure out what to do for certain backbones; do we still need to do this?
        if self.open_ended_generation:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model with flash attention enabled if specified
        # NOTE: `gemma2` models currently do not support flash attention 
        if (not empty_config and not self.config.use_flash_attention) or "gemma" in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        ######################
        # LOAD AUGMENTATIONS #
        ######################
        if not empty_config and "augments" in self.config:
            # Load steering vectors
            if "steering_vector" in self.config.augments:
                # TODO: How to load steering vectors if we already trained?
                if "save_dir" in self.config.augments:
                    pass
                
                else:
                    # Only compute gradients for SV
                    for param in self.model.parameters():
                        param.requires_grad = False

                    for insert_layer in config.insertion_layers:
                        self.model.model.layers[insert_layer].mlp = SteeringHiddenState(
                            self.model.model.layers[insert_layer].mlp,
                            self.model.config.hidden_size,
                            self.config.augment.start_norm
                        )
                    
            # Load LoRA configuration if used
            if "lora" in self.config.augments and self.config.augments.lora.use:
                if "save_dir" in self.config.augments.lora:
                    adapter_config_path = os.path.join(self.config.augments.lora.save_dir, "adapter_config.json")
                    with open(adapter_config_path, "r") as f:
                        adapter_config = json.load(f)

                    lora_config = LoraConfig(
                        r=adapter_config["r"],
                        lora_alpha=adapter_config["lora_alpha"],
                        lora_dropout=adapter_config["lora_dropout"],
                        target_modules=adapter_config["target_modules"],
                        bias=adapter_config["bias"]
                    )

                    # Apply LoRA to the model
                    self.model = get_peft_model(self.model, lora_config)
                    self.model.load_adapter(self.config.augments.lora.save_dir, adapter_name="lora_adapter", device_map="auto")
                else:
                    lora_config = LoraConfig(
                        r=self.config.augments.lora.r,
                        lora_alpha=self.config.augments.lora.alpha,
                        lora_dropout=self.config.augments.lora.dropout,
                        target_modules=self.config.augments.lora.target_modules,
                        bias=self.config.augments.lora.bias
                    )
                    self.model = get_peft_model(self.model, lora_config)

    def infer(self, input_prompt, max_new_tokens=None, temperature=0, infer_method=None):        
        messages = [{"role": "user", "content": input_prompt}]
        input_tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        
        if infer_method is not None:
            if infer_method["generation_type"] == "nucleus":
                output = self.model.generate(
                    input_ids=input_tokens,
                    pad_token_id=self.tokenizer.eos_token_id if self.open_ended_generation else None,
                    do_sample=True,
                    top_p=infer_method["p"],
                    max_new_tokens=max_new_tokens if max_new_tokens!=None else self.config.max_new_tokens)
                
            elif infer_method["generation_type"] == "top_k":
                output = self.model.generate(
                    input_ids=input_tokens,
                    pad_token_id=self.tokenizer.eos_token_id if self.open_ended_generation else None,
                    do_sample=True,
                    top_k=infer_method["k"],
                    max_new_tokens=max_new_tokens if max_new_tokens!=None else self.config.max_new_tokens)
                
            elif infer_method["generation_type"] == "hybrid":
                output = self.model.generate(
                    input_ids=input_tokens,
                    pad_token_id=self.tokenizer.eos_token_id if self.open_ended_generation else None,
                    do_sample=True,
                    top_p=infer_method["p"],
                    temperature=temperature,
                    repetition_penalty=infer_method["repetition_penalty"],
                    no_repeat_ngram_size=infer_method["no_repeat_ngram_size"],
                    max_new_tokens=max_new_tokens if max_new_tokens!=None else self.config.max_new_tokens)
            
            elif infer_method["generation_type"] == "standard":
                if temperature == 0:
                    output = self.model.generate(input_tokens,
                                                pad_token_id=self.tokenizer.eos_token_id if self.open_ended_generation else None,
                                                max_new_tokens=max_new_tokens if max_new_tokens!=None else self.config.max_new_tokens)

                else:
                    output = self.model.generate(input_tokens,
                                                pad_token_id=self.tokenizer.eos_token_id if self.open_ended_generation else None,
                                                do_sample=True,
                                                temperature=temperature,
                                                max_new_tokens=max_new_tokens if max_new_tokens!=None else self.config.max_new_tokens)        
                
            else:
                raise NotImplementedError(f"Generation type `{infer_method['generation_type']}` not implemented.")
            
        else:
            if temperature == 0:
                output = self.model.generate(input_tokens,
                                             pad_token_id=self.tokenizer.eos_token_id if self.open_ended_generation else None,
                                             max_new_tokens=max_new_tokens if max_new_tokens!=None else self.config.max_new_tokens)

            else:
                output = self.model.generate(input_tokens,
                                             pad_token_id=self.tokenizer.eos_token_id if self.open_ended_generation else None,
                                             do_sample=True,
                                             temperature=temperature,
                                             max_new_tokens=max_new_tokens if max_new_tokens!=None else self.config.max_new_tokens)
        
        # Get generated tokens by cutting everything before length of input token count        
        generated_text = self.tokenizer.decode(output[0][len(input_tokens[0]):], skip_special_tokens=True)
            
        return generated_text

    def infer_batch(self, input_prompts, save_dir=None, max_new_tokens=None, temperature=0, infer_method=None):
        # TODO: Rewrite to actually use batching, but somehow add progress bar?
        if save_dir != None and os.path.exists(save_dir):
            outputs = json.load(open(save_dir, "r"))
        else:
            outputs = []
        
        for text in tqdm(input_prompts[len(outputs):], ncols=80, desc="Batch", total=len(input_prompts)-len(outputs), leave=False):
            output = self.infer(text, max_new_tokens=max_new_tokens, temperature=temperature, infer_method=infer_method)
            
            outputs.append(output)
            
            if save_dir != None:
                json.dump(outputs, open(save_dir, "w"), indent=4)

        return outputs
