import torch
from summac.model_summac import SummaCConv


class SummaC_NLI:
    def __init__(self):
        self.model = SummaCConv(
        models=["vitc"],
        bins="percentile",
        granularity="sentence",
        nli_labels="e",
        device=torch.device("cuda"),
    )
    
    def score(self, source, generated):
        return self.model.score([source], [generated])["scores"][0]
        