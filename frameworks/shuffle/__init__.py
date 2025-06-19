import os
import json
import pickle as pkl

from tqdm import tqdm

from utils import STANCES, POLITICAL_STANCES


def get_all_cycles(lst):
    """Generate all cyclic permutations of a list.

    Args:
        lst (list): The list to generate cycles from.

    Returns:
        list: A list containing all cyclic permutations of the input list.
    """
    cycles = []
    n = len(lst)
    for i in range(n):
        cycle = lst[i:] + lst[:i]
        cycles.append(cycle)
    return cycles

class Shuffle:
    """Class to perform inference on shuffled data using a given model."""

    def __init__(self, model):
        """Initialize the Shuffle class.

        Args:
            model: The model used for inference.
        """
        template_path = os.path.join(os.path.dirname(__file__), "prompt_template.pkl")
        with open(template_path, "rb") as f:
            self.template = pkl.load(f)
        self.model = model

    def infer_batch(self, data, save_dir):
        """Perform inference on a batch of data and save the results.

        Args:
            data (list): A list of data instances to process.
            save_dir (str): The path to save the output JSON file.

        Returns:
            None
        """
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        for stance in STANCES:
            stance_cycle_save_dir = save_dir.replace(".json", f"-stance={stance}-cycle.json")
            
            if os.path.exists(stance_cycle_save_dir):
                stance_cycle_outputs = json.load(open(stance_cycle_save_dir))
            else:
                stance_cycle_outputs = []
            
            stance_prompts = []
            
            for instance in data[len(stance_cycle_outputs):]:
                instance_prompts = []
                
                stance_documents = instance[f"{stance}_source"].split("\"\n\"")
                
                for cycle in get_all_cycles(stance_documents):
                    if stance == "A":
                        source_prompt = self.template["source"].format(
                            "\n".join(cycle), instance["B_source"]
                        )
                    else:
                        source_prompt = self.template["source"].format(
                            instance["A_source"], "\n".join(cycle)
                        )

                    prompt = self.template["template"].format(source_prompt).replace(
                        "[STANCE]", POLITICAL_STANCES[stance]
                    )
                    instance_prompts.append(prompt)
                
                stance_prompts.append(instance_prompts)
            
            for cycle_prompts in tqdm(stance_prompts, ncols=80, desc="Inference", leave=False):
                stance_cycle_outputs.append([self.model.infer(stance_prompt) for stance_prompt in cycle_prompts])
                json.dump(stance_cycle_outputs, open(stance_cycle_save_dir, "w"), indent=4)
            
            ########################################
            # COMBINE GENERATED SHUFFLED SUMMARIES #
            ########################################
            stance_save_dir = save_dir.replace(".json", f"-stance={stance}.json")
            
            if os.path.exists(stance_save_dir):
                stance_outputs = json.load(open(stance_save_dir))
            else:
                stance_outputs = []
            
            for stance_cycle_output in stance_cycle_outputs[len(stance_outputs):]:
                prompt = self.template["combine"].format("\n".join(f" - {s}" for s in stance_cycle_output))
                
                stance_outputs.append(self.model.infer(prompt))
                json.dump(stance_outputs, open(stance_save_dir, "w"), indent=4)
            
        #################
        # WRITE TO FILE # 
        #################
        A_data = json.load(open(save_dir.replace(".json", f"-stance=A.json")))
        B_data = json.load(open(save_dir.replace(".json", f"-stance=B.json")))

        for index, instance in enumerate(data):
            data[index]["A_generated"] = A_data[index]
            data[index]["B_generated"] = B_data[index]

        with open(save_dir, "w") as f:
            json.dump(data, f, indent=4)
