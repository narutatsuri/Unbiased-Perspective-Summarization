import os
import yaml
import inspect

from box import Box
from vllm import LLM, SamplingParams


class vLLMModel:
    """
    A class for interacting with a vLLM model for inference.

    This class handles the configuration, initialization, and interaction with the vLLM library.
    It dynamically loads sampling parameters for inference from a configuration file or 
    user-provided configuration and provides methods for single and batch inference.

    Attributes:
        model_name (str): The name of the model being used, extracted from the provided identifier.
        config (Box): A configuration object storing model and inference parameters.
        _sampling_params (SamplingParams): Sampling parameters used during inference.
        model (LLM): An instance of the vLLM model.
    """

    def __init__(self, model_name, config):
        """
        Initializes the vLLMModel with the given model name and configuration.

        Args:
            model_name (str): The identifier for the model. The actual name is extracted from this.
            config (Box or None): Configuration for inference. If None, a default configuration is loaded.

        Raises:
            FileNotFoundError: If the default configuration file is not found.
            yaml.YAMLError: If there is an error parsing the YAML configuration file.
        """
        self.model_name = model_name.split(":")[-1]
        (self.config, empty_config) = (config, False) if config is not None else (Box(), True)
        self.config.update(Box(yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "../config.yaml"), "r"))))

        if not empty_config:
            self._sampling_params = SamplingParams(**{
                key: value
                for key, value in self.config.inference.items()
                if key in set(inspect.signature(SamplingParams).parameters.keys())
            })

        self.model = LLM(model=model_name)

    def infer(self, input_prompt):
        """
        Performs single-prompt inference using the vLLM model.

        Args:
            input_prompt (str): The input text prompt for inference.

        Returns:
            str: The generated output text from the model.
        """
        return self.model.generate(input_prompt, self._sampling_params)[0].outputs[0].text

    def infer_batch(self, input_prompts, save_dir=None):
        """
        Performs batch inference for a list of prompts.

        Args:
            input_prompts (list of str): A list of input text prompts for inference.
            save_dir (str or None): Directory to save the outputs. Currently not used.

        Returns:
            list of str: A list of generated output texts, one for each input prompt.
        """
        outputs = self.model.generate(input_prompts, self._sampling_params)
        return [output.outputs[0].text for output in outputs]
