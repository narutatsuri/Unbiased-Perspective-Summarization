import os
import json
import yaml
import random
from itertools import combinations

import gc
import torch
import wandb
import numpy as np
from box import Box
from tqdm import tqdm
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from utils import get_yaml_hash, set_seeds
from metrics import get_metric


class DPO:
    def __init__(self, model, config_path):
        """
        Initialize the DPO class with the given configuration.
        """
        self.device = torch.device("cuda")
        self.model = model
        self.full_config = Box(yaml.safe_load(open(config_path, "r")))
        
        # If there is a `dpo` parameter, this means we are looking to train the model
        # (otherwise, we only load the model from the passed train path)
        if "dpo" in self.full_config:
            self.config = self.full_config.dpo

            set_seeds(self.config.seed)
            yaml_hash = get_yaml_hash(self.config)
            data_basename = os.path.basename(self.config.data_dir).replace(".json", "")
            config_index = data_basename.index("config=") + len("config=")
            data_config = data_basename[config_index:]
            self.run_name = f'dpo-data={data_config}-sha={yaml_hash}'
            self.save_dir = os.path.join(self.config.save_dir, self.run_name)

            self._train_model()
        else:
            if self.full_config.generation.metrics is not None:
                self.metrics = [get_metric(metric.name, metric.config) for metric in self.full_config.generation.metrics]
            else:
                self.metrics = None

    def infer(self, content, source=None):
        """
        """
        if self.full_config.generation.inference == "rerank":
            assert self.metrics is not None
            
            generations = [self.model.infer(content, temperature=0 if "temperature" not in generation_type else generation_type.temperature, infer_method=generation_type) 
                            for _ in range(self.full_config.generation.num_responses_per_type) 
                            for generation_type in self.full_config.generation.types]
            generation_types = [generation_type for i in range(self.full_config.generation.num_responses_per_type) for generation_type in self.full_config.generation.types]
            
            scores = [self._get_score([metric.score(source, generation) for metric in self.metrics], 
                                    self.full_config.generation.combine_method) for generation in generations]
            
            return generations[np.argmax(scores)], generation_types[np.argmax(scores)]
        
        else:
            assert len(self.full_config.generation.types) == 1
            
            generation_type = self.full_config.generation.types[0]
            return self.model.infer(content, temperature=0 if "temperature" not in generation_type else generation_type.temperature, infer_method=generation_type)
            
    def infer_batch(self, data, save_dir):
        """
        Run inference on the given data and save the generated outputs.

        Args:
            data (list): List of data instances to infer.
            save_dir (str): Directory to save the inferred data.
        """
        if os.path.exists(save_dir):
            inferred_data = json.load(open(save_dir))
        else:
            inferred_data = []

        for instance in tqdm(data[len(inferred_data):], ncols=80, leave=False, desc="DPO"):
            new_instance = dict(instance)
            new_instance["generated"], new_instance["generated_type"] = self.infer(instance["input"], source=instance["document"])
            inferred_data.append(new_instance)
            
            json.dump(inferred_data, open(save_dir, "w"), indent=4)

    def _train_model(self):
        """
        Train the model using the DPO method.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Save configuration
        yaml.dump(self.full_config.to_dict(), open(os.path.join(self.save_dir, "config.yaml"), "w"))

        # Load the dataset
        train_dataset = json.load(open(self.config.data_dir))
        
        # Get list of metrics
        self.metrics = [get_metric(metric.name, metric.config) for metric in self.config.metric.list]

        if self.config.use_wandb:
            wandb.init(dir=".wandb_log", name=self.run_name, config=self.config.to_dict())
        
        # Prepare training arguments
        training_args = DPOConfig(
            output_dir=self.save_dir,
            per_device_train_batch_size=self.config.batch_size if self.config.batch_size is not None else DPOConfig.per_device_train_batch_size,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate if self.config.learning_rate is not None else DPOConfig.learning_rate,
            logging_strategy="steps",
            logging_steps=self.config.log_interval,
            save_steps=self.config.eval_interval,
            save_total_limit=1,
            report_to="wandb" if self.config.use_wandb else None,
            run_name=self.run_name,
        )

        # Initialize the trainer
        # NOTE: We pass an empty trainer here but update it in the loop each step
        trainer = None

        train_data = []

        for iteration in range(self.config.num_iterations):
            new_data = self._generate_new_data(
                iteration,
                self.save_dir,
                train_dataset,
                self.model,
                self.metrics
            )
            train_data.extend(new_data)

            # remove the old trainer instance
            if trainer is not None:
                del trainer  
                gc.collect()
                torch.cuda.empty_cache()

            # Convert new_data to Dataset instance
            dataset_dict = {key: [d[key] for d in new_data] for key in new_data[0]}
            new_dataset = Dataset.from_dict(dataset_dict)
            
            # Initialize the trainer            
            trainer = DPOTrainer(
                model=self.model.model,
                tokenizer=self.model.tokenizer,
                train_dataset=new_dataset,
                eval_dataset=None,
                args=training_args
            )

            trainer.train()

            # Log any iteration-specific metrics or information to the same WandB run
            if self.config.use_wandb:
                wandb.log({"iteration": iteration + 1, "num_samples": len(train_data)})

            # Save model after each iteration
            iteration_save_dir = os.path.join(self.save_dir, f"iteration={iteration + 1}")
            self.model.model.save_pretrained(iteration_save_dir)
            self.model.tokenizer.save_pretrained(iteration_save_dir)

            # Save training data after each iteration
            with open(os.path.join(self.save_dir, f"train_data-iteration={iteration + 1}.json"), "w") as f:
                json.dump(train_data, f)

        # Finish WandB run if used
        if self.config.use_wandb:
            wandb.finish()

        # Save the trained model
        self.model.model.save_pretrained(self.save_dir)
        self.model.tokenizer.save_pretrained(self.save_dir)

    def _generate_new_data(self, iteration, save_dir, base_data, model, metrics):
        """
        Generate new training data by creating pairs of model responses and their preference scores.

        Args:
            iteration (int): Current iteration number.
            save_dir (str): Directory to save temporary generated data.
            base_data (list): List of base data instances.
            model: The model used for generation.
            metrics (list): The metrics used to score the responses.
            subsample (int, optional): Number of samples to subsample from base_data. Defaults to 500.

        Returns:
            list: A list of new data instances with "prompt", "chosen", and "rejected" keys.
        """
        temp_save_dir = os.path.join(save_dir, "temp_generated_data")

        if not os.path.exists(temp_save_dir):
            os.makedirs(temp_save_dir)

        # Sample subset of full data to use to generate training data
        if self.config.data_config.subsample_base.use:
            data = random.sample(base_data, k=self.config.data_config.subsample_base.count)
        else:
            data = base_data

        temp_file = os.path.join(temp_save_dir, f"iteration={iteration + 1}.json")
        if os.path.exists(temp_file):
            new_data_log = json.load(open(temp_file))

            new_data = []
            for instance in [j for i in new_data_log for j in i]:
                new_instance = dict(instance)
                del new_instance["chosen_score"]
                del new_instance["rejected_score"]
                new_data.append(new_instance)
        else:
            new_data_log = []
            new_data = []

        for instance in tqdm(data[len(new_data_log):], ncols=80, total=len(data)-len(new_data_log), desc="Generating Data", leave=False):
            input_text = instance["input"]
            rm_input_text = instance["document"]

            generations = [model.infer(input_text, 
                                       temperature=0 if "temperature" not in generation_type else generation_type.temperature, 
                                       infer_method=generation_type) 
                           for _ in range(self.config.generations.num_responses_per_type) 
                           for generation_type in self.config.generations.types]

            # Score generation
            scores = [self._get_score([metric.score(rm_input_text, generation) 
                                       for metric in metrics], 
                                      self.config.metric.combine_method) 
                      for generation in generations]

            ordered_index_pairs = [
                (i, j) if scores[i] > scores[j] else (j, i)
                for i, j in combinations(range(len(scores)), 2)
                if scores[i] != scores[j]  # Only include pairs with unequal scores
            ]
            
            if self.config.data_config.train_best.use:
                ordered_index_pairs.sort(
                    key=lambda pair: abs(scores[pair[0]] - scores[pair[1]]),
                    reverse=True
                )
                ordered_index_pairs = ordered_index_pairs[:self.config.data_config.train_best.count]
            
            instance_batch = []
            for index_pair in ordered_index_pairs:
                new_data.append({
                    "prompt": input_text,
                    "chosen": generations[index_pair[0]],
                    "rejected": generations[index_pair[1]]
                })

                instance_batch.append({
                    "prompt": input_text,
                    "chosen": generations[index_pair[0]],
                    "rejected": generations[index_pair[1]],
                    "chosen_score": scores[index_pair[0]],
                    "rejected_score": scores[index_pair[1]]
                })
                
            new_data_log.append(instance_batch)

            json.dump(new_data_log, open(temp_file, "w"), indent=4)

        return new_data

    def _get_score(self, scores, combine_method):
        if combine_method == "average":
            return np.mean(scores)
        
        else:
            raise NotImplementedError()