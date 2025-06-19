import os
import yaml
from typing import List, Literal

import numpy as np
from box import Box
from rouge_score import rouge_scorer


class RougeScore:
    def __init__(self, config):
        """
        Initializes the RougeScore object.

        Parameters:
            config (dict): Configuration dictionary with the following keys:
                - rouge_types (List[str]): List of ROUGE types to use.
                  Options are:
                    - "rouge1"
                    - "rouge2"
                    - "rougeL"
        
        Raises:
            ValueError: If any rouge_type in config is not supported.
        """
        # Define acceptable values for rouge_types
        self.config = Box(yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r")))
        if config is not None:
            for param in config:
                self.config[param] = config[param]
                        
        allowed_rouge_types: List[Literal["rouge1", "rouge2", "rougeL"]] = self.config.rouge_types

        # Ensure all provided rouge types are valid
        for rouge_type in allowed_rouge_types:
            if rouge_type not in ["rouge1", "rouge2", "rougeL"]:
                raise ValueError(f"Unsupported rouge_type: {rouge_type}")

        self.rouge_types = allowed_rouge_types
        self.model = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=False)
    
    def score(self, reference, generated):
        """
        Computes ROUGE scores.

        Parameters:
            reference (str): The reference text.
            generated (str): The generated text to compare.

        Returns:
            dict: A dictionary with ROUGE types as keys and their corresponding scores.
        """
        score = self.model.score(reference, generated)
        final_scores = {}
        
        for rouge_type in self.rouge_types:
            if self.config.measure == "f1":
                final_scores[rouge_type] = score[rouge_type].fmeasure
            elif self.config.measure == "precision":
                final_scores[rouge_type] = score[rouge_type].precision
            elif self.config.measure == "recall":
                final_scores[rouge_type] = score[rouge_type].recall 
            elif self.config.measure == "all":
                final_scores[f"{rouge_type}_f1"] = score[rouge_type].fmeasure
                final_scores[f"{rouge_type}_precision"] = score[rouge_type].precision
                final_scores[f"{rouge_type}_recall"] = score[rouge_type].recall

        if self.config.combine is None:
            return final_scores
        elif self.config.combine == "average":
            return np.mean(list(final_scores.values()))