import os
import json
import yaml
import pickle as pkl

from box import Box
from tqdm import tqdm

from utils import STANCES, print_status


class MultiAgentDebate:
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
        
        # We set the response of the first agent to the answer by the baseline
        if not os.path.exists(step_0_save_dir):
            
            responses = []
            for agent_prompts in tqdm([[instance[f"input"]] * self.config.agents for instance in data], ncols=80, leave=False, desc="Round 0"):
                response = []
                for prompt in agent_prompts:
                    response.append(self.model.infer(prompt, temperature=self.config.temperature))
                responses.append(response)

            for index, response in enumerate(responses):
                data[index][f"generated_step_0"] = response

            json.dump(data, open(step_0_save_dir, "w"), indent=4)
        else:
            data = json.load(open(step_0_save_dir, "r"))

        ##########
        # DEBATE #
        ##########
        for round in range(1, self.config.rounds + 1):
            step_save_dir = save_dir.replace(".json", f"-step_{round}.json")

            if not os.path.exists(step_save_dir):
                print_status(f"Debate: Round {round}")

                prompts = []

                for instance in data:
                    instance_prompt = []

                    for agent_index in range(self.config.agents):
                        agent_response = instance[f"generated_step_{round-1}"][agent_index]
                        other_responses = [instance[f"generated_step_{round-1}"][other_index] for other_index in range(self.config.agents) if other_index != agent_index]

                        instance_prompt.append(self.template["agent_refine"].format(instance[f"input"],
                                                                                    agent_response,
                                                                                    " - " + "\n - ".join(other_responses)))

                    prompts.append(instance_prompt)

                responses = [[self.model.infer(prompt, temperature=self.config.temperature) 
                            for prompt in agent_prompts] 
                            for agent_prompts in prompts]

                for index, response in enumerate(responses):
                    data[index][f"generated_step_{round}"] = response

                json.dump(data, open(step_save_dir, "w"), indent=4)
            else:
                data = json.load(open(step_save_dir, "r"))

        ###########################
        # (OPTIONAL) FINAL ANSWER #
        ###########################
        # If parameter is set to True, generate one final output based on latest round
        if not os.path.exists(save_dir):
            if self.config.final_agent:
                print_status(f"Debate: Final Round")
                prompts = [
                    self.template["final_refine"].format(
                        instance[f"input"],
                        " - " + "\n - ".join(instance[f"generated_step_{round}"]),
                    )
                    for instance in data
                ]

                responses = [[self.model.infer(prompt, temperature=self.config.temperature) 
                            for prompt in agent_prompts] 
                            for agent_prompts in prompts]

                for index, response in enumerate(responses):
                    data[index][f"generated_summary"] = response

            else:
                for instance in data:
                    # Arbitrarily pick summary of first agent as final output
                    # TODO: Can this be improved?
                    instance[f"generated_summary"] = instance[f"generated_step_{round}"][0]

            json.dump(data, open(save_dir, "w"), indent=4)
