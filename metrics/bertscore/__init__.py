import os
import yaml

import torch
from box import Box
from torchmetrics.text.bert import BERTScore


class BERT:
    def __init__(self, config):
        self.model = BERTScore(model_name_or_path="microsoft/deberta-large-mnli", device=torch.device("cuda"))
        
        self.config = Box(yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r")))
        if config is not None:
            for param in config:
                self.config[param] = config[param]
    
    def score(self, reference, generated):
        return self.model(preds=[generated], target=[reference])[self.config.bert_type].item()