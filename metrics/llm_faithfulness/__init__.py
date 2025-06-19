import os
import re
import yaml

import nltk
import rouge
import torch
import tiktoken
import numpy as np
from box import Box
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.cross_encoder import CrossEncoder
from tenacity import retry, stop_after_attempt, retry_if_result

from utils import openai_context, remove_numbered_lists, print_error
from backbones import get_backbone


def extract_number(text):
    # Search for the first occurrence of a number in the string
    match = re.search(r'\d+', text)
    if match:
        return float(match.group())  # Convert the matched string to an integer
    return None

class LLMFaithfulness:
    def __init__(self, config):
        self.name = self.__class__.__name__
        self.config = Box(yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8")))
        if config is not None:
            for param in config:
                self.config[param] = config[param]
            
        self.prompt = getattr(self.config, f"prompt_{self.config.prompt_mode}", None)
            
        # Initialize models and necessary components
        self.budget = self.config.budget  # Set the budget
        self.device = "cuda"

        if self.config.extract:
            # BERTScore model
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model)
            self.bert_model = AutoModel.from_pretrained(self.config.bert_model).to(self.device).eval()

            # NLI Model
            self.nli_model = CrossEncoder(self.config.nli_model, max_length=512, device=self.device)

            # ROUGE Evaluator
            self.rouge_evaluator = rouge.Rouge(
                metrics=["rouge-n"],
                max_n=2,
                limit_length=False,
                apply_avg=False,
                stemming=True,
                ensure_compatibility=True,
            )

        # Tokenizer
        if "hf:" in self.config.prompt_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.prompt_model.split(":")[-1])
        elif "openai:" in self.config.prompt_model:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
            
        self.backbone = get_backbone(self.config.prompt_model)
        
    def _extract_bertscore(self, source, generated):
        # BERTScore-based extraction
        src = nltk.sent_tokenize(source.replace("<br />", "\n"))

        with torch.no_grad():
            # System output embedding
            batch_sys = self.bert_tokenizer(generated, return_tensors="pt", truncation=True).to(self.device)
            generated_emb = self.bert_model(**batch_sys).last_hidden_state.mean(dim=1).squeeze(0)
            generated_emb = generated_emb / generated_emb.norm()

            # Source sentence embeddings
            batch_src = self.bert_tokenizer(src, padding=True, truncation=True, return_tensors="pt").to(self.device)
            src_sent_emb = self.bert_model(**batch_src).last_hidden_state.mean(dim=1)
            src_sent_emb = src_sent_emb / src_sent_emb.norm(dim=1, keepdim=True)

            # Similarities
            sim = src_sent_emb @ generated_emb
            sim = sim.cpu().numpy()

        return self._select_sentences(src, sim)

    def _extract_lead(self, source):
        # Lead-based extraction
        src = nltk.sent_tokenize(source.replace("<br />", "\n"))

        extracted = []
        num_tokens = 0
        enc = self.tokenizer

        for s in src:
            s_tokens = len(enc.encode(s)) + 1  # +1 for separator token
            if num_tokens + s_tokens > self.budget:
                break
            extracted.append(s)
            num_tokens += s_tokens
            
        return " ".join(extracted)

    def _extract_nli(self, source, generated):
        # NLI-based extraction
        src = nltk.sent_tokenize(source.replace("<br />", "\n"))
        premises = nltk.sent_tokenize(generated)

        inputs = [(premise, hypothesis) for hypothesis in src for premise in premises]
        probs = self.nli_model.predict(inputs, apply_softmax=True)  # Shape: (num_samples, 3)

        # Reshape probabilities to (num_hypotheses, num_premises, 3)
        num_hypotheses = len(src)
        num_premises = len(premises)
        probs = probs.reshape(num_hypotheses, num_premises, 3)

        # Extract entailment probabilities (assuming index 0 corresponds to entailment)
        entailment_probs = probs[:, :, 0]

        # Maximum entailment probability for each hypothesis
        max_entailment_probs = entailment_probs.max(axis=1)
        scores = max_entailment_probs
        
        return self._select_sentences(src, scores)

    def _extract_rouge(self, source, generated):
        # ROUGE-based extraction
        src = nltk.sent_tokenize(source.replace("<br />", "\n"))
        cand_doc = src  # Each sentence is a candidate

        scores = self.rouge_evaluator.get_scores(cand_doc, [generated] * len(cand_doc))
        # Access ROUGE-1 recall scores
        rouge_scores = [score["r"] for score in scores["rouge-1"]]
        
        return self._select_sentences(src, np.array(rouge_scores).flatten())

    def _select_sentences(self, src, scores):
        # Helper function to select sentences based on scores and the token budget
        extracted_src = []
        index = np.argsort(scores)[::-1]
        num_tokens = 0
        enc = self.tokenizer

        for i in index:
            s = src[i]
            s_tokens = len(enc.encode(s)) + 1  # +1 for separator token
            if num_tokens + s_tokens > self.budget:
                continue
            extracted_src.append(s)
            num_tokens += s_tokens
            if num_tokens >= self.budget:
                break

        return " ".join(extracted_src)

    def score(self, source, generated):
        # Extract sentences from source based on the specified method
        if self.config.extract:
            article = " ".join([self._extract_bertscore(source, generated),
                                self._extract_lead(source),
                                self._extract_nli(source, generated),
                                self._extract_rouge(source, generated)])
        else:
            article = source
        
        @retry(stop=stop_after_attempt(3), 
                retry=retry_if_result(lambda result: result is None), 
                retry_error_callback=lambda retry_state: 1)
        def get_response():
            task_prompt = self.prompt.replace("{{article}}", article).replace("{{summary}}", generated)

            # Check context length; if too long, 
            tokenized_prompt = self.tokenizer.encode(task_prompt)
            
            max_context = openai_context.get(self.config.prompt_model.split(":")[-1], 
                                             {}).get("context", 
                                                     self.backbone.model.config.max_position_embeddings)

            if len(tokenized_prompt) >= max_context:
                raise OverflowError("Input prompt too long.")
            
            # Call inference backbone
            response = self.backbone.infer(task_prompt)
            
            # Process response string (remove numbered lists, etc.)
            response = remove_numbered_lists(response)

            # Validate and process response
            try:
                return float(response)
            except ValueError:
                return extract_number(response)
        
        return get_response()