import os

import torch
from alignscore import AlignScore

from utils import BASE_METRICS_DIR


class Align:
    def __init__(self):
        self.model = AlignScore(
        model="roberta-base",
        batch_size=32,
        device=torch.device("cuda"),
        evaluation_mode="nli_sp",
        ckpt_path=os.path.join(BASE_METRICS_DIR, "AlignScore-large.ckpt"),
        verbose=False,
    )
    
    def score(self, source, generated):
        return self.model.score(contexts=[source], claims=[generated])[0]