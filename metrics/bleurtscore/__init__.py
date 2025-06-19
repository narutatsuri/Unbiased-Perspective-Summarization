import os

from bleurt import score

from utils import BASE_METRICS_DIR


class BLEURTScore:
    def __init__(self):
        self.model = score.BleurtScorer(os.path.join(BASE_METRICS_DIR, "BLEURT-20-D6"))
    
    def score(self, reference, generated):
        return self.model.score(references=[reference], candidates=[generated])[0]