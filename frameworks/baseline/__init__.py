import os
import json
import yaml

import numpy as np
from box import Box
from tqdm import tqdm

from metrics import get_metric


class Baseline:
    def __init__(self, model, config_path):
        self.config = Box(yaml.safe_load(open(config_path, "r")))        
        self.model = model
        
        if self.config.generation.method is not None:
            self.metrics = [get_metric(metric.name, metric.config) for metric in self.config.generation.metric.list]

    def infer_batch(self, data, save_dir):
        if not os.path.exists(os.path.dirname(save_dir)) and os.path.dirname(save_dir) != "":
            os.makedirs(os.path.dirname(save_dir))
        
        if os.path.exists(save_dir):
            new_data = json.load(open(save_dir))
        else:
            new_data = []
            
        for instance in tqdm(data[len(new_data):], desc="Baseline", ncols=80, leave=False):
            new_instance = dict(instance)
            
            if self.config.generation.method == "rerank":
                generations = [self.model.infer(instance["input"], temperature=0 if "temperature" not in generation_type else generation_type.temperature, infer_method=generation_type) 
                                for _ in range(self.config.generation.num_responses_per_type) 
                                for generation_type in self.config.generation.types]
                                
                scores = [self._get_score([metric.score(instance["document"], generation) for metric in self.metrics], 
                                        self.config.generation.metric.combine_method) for generation in generations]
                
                new_instance["generated"] = generations[np.argmax(scores)]
            
            else:
                new_instance["generated"] = self.model.infer(instance["input"])
                
            new_data.append(new_instance)
            
            json.dump(new_data, open(save_dir, "w"), indent=4)
            
    def _get_score(self, scores, combine_method):
        if combine_method == "average":
            return np.mean(scores)
        
        else:
            raise NotImplementedError()            