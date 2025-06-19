import os
import json
import yaml
import pickle as pkl

from box import Box

from utils import STANCES, print_status


class SelfRefine:
    def __init__(self, model, config_path):
        self.config = Box(yaml.safe_load(open(config_path, "r")))
        self.template = pkl.load(open(os.path.join(os.path.dirname(__file__), "prompt_template.pkl"), "rb"))

        self.model = model

    def infer_batch(self, data, save_dir):
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        ################
        # ZEROTH ROUND #
        ################
        step_0_save_dir = save_dir.replace(".json", "-step_0.json")

        if not os.path.exists(step_0_save_dir):
            stance_prompts = [instance[f"input"] for instance in data]

            responses = self.model.infer_batch(
                stance_prompts,
                step_0_save_dir,
                temperature=self.config.temperature,
            )

            for index, response in enumerate(responses):
                data[index][f"generated_step_0"] = response

            json.dump(data, open(step_0_save_dir, "w"), indent=4)
        else:
            data = json.load(open(step_0_save_dir, "r"))

        for round in range(1, self.config.rounds + 1):
            step_save_dir = save_dir.replace(".json", f"-step_{round}.json")

            if not os.path.exists(step_save_dir):
                print_status(f"Self-Refine: Round {round}")
                
                ############
                # FEEDBACK #
                ############
                feedback_step_save_dir = step_save_dir.replace(".json", f"-feedback.json")

                feedback_prompts = [self.template["feedback"].format(instance[f"input"], instance[f"generated_step_{round-1}"]) for instance in data]
                
                responses = self.model.infer_batch(feedback_prompts,
                                                    feedback_step_save_dir,
                                                    temperature=self.config.temperature)

                for index, response in enumerate(responses):
                    data[index][f"generated_step_{round}_feedback"] = response

                json.dump(data, open(feedback_step_save_dir, "w"), indent=4)
                
                ##########
                # REFINE #
                ##########
                refine_step_save_dir = step_save_dir.replace(".json", f"-refine.json")

                refine_prompts = []
                
                for instance in data:
                    refine_prompt = self.template["feedback"].format(instance[f"input"], instance[f"generated_step_{round-1}"])
                    refine_prompt += self.template["refine"].format(instance[f"generated_step_{round}_feedback"])
                    
                    refine_prompts.append(refine_prompt)
                                        
                responses = self.model.infer_batch(refine_prompts,
                                                    refine_step_save_dir,
                                                    temperature=self.config.temperature)

                for index, response in enumerate(responses):
                    data[index][f"generated_step_{round}"] = response

                json.dump(data, open(feedback_step_save_dir, "w"), indent=4)
                
            else:
                data = json.load(open(step_save_dir, "r"))

        ################
        # FINAL ANSWER #
        ################
        if not os.path.exists(save_dir):
            for instance in data:
                instance[f"generated"] = instance[
                    f"generated_step_{round}"
                ]
                    
            json.dump(data, open(save_dir, "w"), indent=4)
