import os
import sys


def get_backbone(model_name, config=None):
    """
    Load and return the appropriate model object based on the given model name.

    This function dynamically adds the current directory to the system path,
    checks the provided model name, and imports and initializes the corresponding
    model class.

    Parameters:
    model (str): The name of the model to load. Expected values are substrings 
                 that indicate the type of model, such as "openai" for OpenAI models 
                 or other strings for local Llama2 models.

    Returns:
    object: An instance of the specified model class, either OpenAIModel or Llama2Model.
    """
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))

    if "openai:" in model_name:
        if config is not None:
            raise ValueError("Parameter `config` should be uninitialized for OpenAI model")
        
        from openai_model import OpenAIModel
        model = OpenAIModel(model_name)

    elif "hf:" in model_name:
        from hf_model import HFModel
        model = HFModel(model_name, config)

    elif "vllm:" in model_name:
        from vllm_model import vLLMModel
        model = vLLMModel(model_name, config)

    else:
        raise NotImplementedError              
        
    return model