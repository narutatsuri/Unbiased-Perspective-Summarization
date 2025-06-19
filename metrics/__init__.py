import os
import sys


def get_metric(
    metric,
    config=None):
    """
    Returns a metric object based on the metric name provided.

    Parameters:
        metric (str): The name of the metric. 

    Returns:
        An instance of the selected metric class.

    Raises:
        ValueError: If 'rouge' is selected and 'config_path' is not provided.
        NotImplementedError: If an unsupported metric is requested.
    """
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    if metric == "rouge":
        from rouge_scorer import RougeScore

        return RougeScore(config)

    elif metric == "bertscore":
        from bertscore import BERT

        return BERT(config)

    elif metric == "bleurtscore":
        from bleurtscore import BLEURTScore

        return BLEURTScore()

    elif metric == "alignscore":
        from align import Align

        return Align()

    elif metric == "summac":
        from summac_nli import SummaC_NLI

        return SummaC_NLI()

    # LLM-based metrics
    elif metric == "extract_then_evaluate":
        from extract_then_evaluate import ExtractThenEvaluate

        return ExtractThenEvaluate()

    elif metric == "llm_coverage":
        from llm_coverage import LLMCoverage

        return LLMCoverage(config)
    
    elif metric == "llm_faithfulness":
        from llm_faithfulness import LLMFaithfulness

        return LLMFaithfulness(config)
    
    elif metric == "unieval":
        from unieval import UniEval

        return UniEval()    

    elif metric == "finesure":
        from finesure import FineSure

        return FineSure()

    elif metric == "minicheck":
        from minicheck import MiniCheck

        return MiniCheck()        

    else:
        raise NotImplementedError(f"Metric '{metric}' not implemented")
