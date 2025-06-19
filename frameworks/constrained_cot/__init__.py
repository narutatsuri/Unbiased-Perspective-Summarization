import os
import json
import pickle as pkl

from utils import STANCES, POLITICAL_STANCES, print_status


class ConstrainedCoT:
    def __init__(self, model):
        self.template = pkl.load(open(os.path.join(os.path.dirname(__file__), "prompt_template.pkl"), "rb"))

        self.model = model

    def infer_batch(self, data, save_dir):
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        ########
        # LIST #
        ########
        list_save_dir = save_dir.replace(".json", "-list.json")

        if not os.path.exists(list_save_dir):
            print_status(f"Constrained CoT: List")
            
            for stance in STANCES:
                stance_prompts = [self.template["list"].format(self.template["source"].format(instance["A_prompt"], instance["B_prompt"])).replace("[STANCE]", POLITICAL_STANCES[stance]) for instance in data]

                responses = self.model.infer_batch(stance_prompts, list_save_dir.replace(".json", f"-stance={stance}.json"),)

                for index, response in enumerate(responses):
                    data[index][f"{stance}_list"] = response

            json.dump(data, open(list_save_dir, "w"), indent=4)
        else:
            data = json.load(open(list_save_dir, "r"))

        #############
        # SUMMARIZE #
        #############
        summarize_save_dir = save_dir.replace(".json", "-summarize.json")
        
        if not os.path.exists(summarize_save_dir):
            print_status(f"Constrained CoT: Summarize")

            for stance in STANCES:
                stance_prompts = []
                for instance in data:
                    list_prompt = self.template["list"].format(self.template["source"].format(instance["A_prompt"], instance["B_prompt"])).replace("[STANCE]", POLITICAL_STANCES[stance])
                    stance_prompts.append(self.template["summarize"].format(list_prompt, instance[f"{stance}_list"]).replace("[STANCE]", POLITICAL_STANCES[stance]))
                
                responses = self.model.infer_batch(stance_prompts, summarize_save_dir.replace(".json", f"-stance={stance}.json"),)

                for index, response in enumerate(responses):
                    data[index][f"{stance}_summarize"] = response

            json.dump(data, open(summarize_save_dir, "w"), indent=4)
        else:
            data = json.load(open(summarize_save_dir, "r"))

        ############
        # CRITIQUE #
        ############
        critique_save_dir = save_dir.replace(".json", "-critique.json")

        if not os.path.exists(critique_save_dir):
            print_status(f"Constrained CoT: Critique")

            for stance in STANCES:
                stance_prompts = []
                for instance in data:
                    list_prompt = self.template["list"].format(self.template["source"].format(instance["A_prompt"], instance["B_prompt"])).replace("[STANCE]", POLITICAL_STANCES[stance])
                    summarize_prompt = self.template["summarize"].format(list_prompt, instance[f"{stance}_list"]).replace("[STANCE]", POLITICAL_STANCES[stance])
                    stance_prompts.append(self.template["critique"].format(summarize_prompt, instance[f"{stance}_summarize"]))
                
                responses = self.model.infer_batch(stance_prompts, critique_save_dir.replace(".json", f"-stance={stance}.json"),)

                for index, response in enumerate(responses):
                    data[index][f"{stance}_critique"] = response

            json.dump(data, open(critique_save_dir, "w"), indent=4)
        else:
            data = json.load(open(critique_save_dir, "r"))            
            
        ##########
        # REVISE #
        ##########
        if not os.path.exists(save_dir):
            print_status(f"Constrained CoT: Revise")

            for stance in STANCES:
                stance_prompts = []
                for instance in data:
                    list_prompt = self.template["list"].format(self.template["source"].format(instance["A_prompt"], instance["B_prompt"])).replace("[STANCE]", POLITICAL_STANCES[stance])
                    summarize_prompt = self.template["summarize"].format(list_prompt, instance[f"{stance}_list"]).replace("[STANCE]", POLITICAL_STANCES[stance])
                    critique_prompt = self.template["critique"].format(summarize_prompt, instance[f"{stance}_summarize"])
                    revise_prompt = self.template["revise"].format(critique_prompt, instance[f"{stance}_critique"]).replace("[STANCE]", POLITICAL_STANCES[stance])
                                        
                    # revise_prompt = self.template["revise_no_document"].format(instance[f"{stance}_list"], 
                    #                                                            instance[f"{stance}_summarize"], 
                    #                                                            instance[f"{stance}_critique"]).replace("[STANCE]", POLITICAL_STANCES[stance])
                    
                    stance_prompts.append(revise_prompt)
                
                responses = self.model.infer_batch(stance_prompts, save_dir.replace(".json", f"-stance={stance}.json"),)

                for index, response in enumerate(responses):
                    data[index][f"{stance}_generated"] = response

            json.dump(data, open(save_dir, "w"), indent=4)
