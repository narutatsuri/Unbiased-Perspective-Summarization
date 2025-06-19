import os
import sys
import yaml

from box import Box

from backbones import get_backbone


def get_framework(framework, config_path):
    
    config = Box(yaml.safe_load(open(config_path, "r")))
    
    if "backbone" in config:
        if "model" not in config.backbone:
            raise ValueError("Backbone model not specified in framework config")
        
        model = get_backbone(config.backbone.model, config.backbone.params) if "params" in config.backbone else get_backbone(config.backbone.model)
    else:
        raise ValueError("Parameter 'backbone' not found in framework config")

    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    if framework == "baseline": 
        from baseline import Baseline
        return Baseline(model, config_path)

    elif framework == "debate":
        from debate import MultiAgentDebate
        return MultiAgentDebate(model, config_path)

    elif framework == "self_refine":
        from self_refine import SelfRefine
        return SelfRefine(model, config_path)

    elif framework == "cot":
        from constrained_cot import ConstrainedCoT
        return ConstrainedCoT(model)

    elif framework == "shuffle":
        from shuffle import Shuffle
        return Shuffle(model) 

    elif framework == "pine":
        from attention_pine import PINE
        return PINE(config_path)
    
    elif framework == "dpo":
        from dpo import DPO
        return DPO(model, config_path)
    
    else:
        raise NotImplementedError