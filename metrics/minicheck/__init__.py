from typing import List

import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


SYSTEM_PROMPT = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. Please assess the claim's consistency with the document by responding with either "Yes" or "No"."""

USER_PROMPT = """Document: [DOCUMENT]\nClaim: [CLAIM]"""

class MiniCheck:
    def __init__(self, model_name='flan-t5-large', max_model_len=None, batch_size=16) -> None:

        self.model = Inferencer(
            model_name=model_name, 
            batch_size=batch_size, 
            max_model_len=max_model_len,
        )

    def score(self, source, generated) -> List[float]:
        self.model.chunk_size = 500 if self.model.model_name == 'flan-t5-large' else 400

        max_support_prob, _, _ = self.model.fact_check([source], [generated])

        return max_support_prob[0]

def sent_tokenize_with_newlines(text):
    blocks = text.split('\n')
    
    tokenized_blocks = [sent_tokenize(block) for block in blocks]
    tokenized_text = []
    for block in tokenized_blocks:
        tokenized_text.extend(block)
        tokenized_text.append('\n')  

    return tokenized_text[:-1]  


class Inferencer():
    def __init__(self, model_name, max_model_len, batch_size) -> None:
        
        self.model_name = model_name

        ckpt = 'lytang/MiniCheck-Flan-T5-Large'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)

        self.max_model_len=2048 if max_model_len is None else max_model_len
        self.max_output_length = 256

        self.model.eval()
        self.batch_size = batch_size
        self.softmax = nn.Softmax(dim=-1)

    def inference_example_batch(self, doc: list, claim: list):
        """
        inference a example,
        doc: list
        claim: list
        using self.inference to batch the process
        """

        assert len(doc) == len(claim), "doc must has the same length with claim!"

        max_support_probs = []
        used_chunks = []
        support_prob_per_chunk = []
        
        for one_doc, one_claim in zip(doc, claim):
            output = self.inference_per_example(one_doc, one_claim)
            max_support_probs.append(output['max_support_prob'])
            used_chunks.append(output['used_chunks'])
            support_prob_per_chunk.append(output['support_prob_per_chunk'])
        
        return {
            'max_support_probs': max_support_probs,
            'used_chunks': used_chunks,
            'support_prob_per_chunk': support_prob_per_chunk
        }

    def inference_per_example(self, doc:str, claim: str):
        """
        inference a example,
        doc: string
        claim: string
        using self.inference to batch the process
        """
        def chunks(lst, n):
            """Yield successive chunks from lst with each having approximately n tokens.

            For flan-t5, we split using the white space;
            For roberta and deberta, we split using the tokenization.
            """
            if self.model_name == 'flan-t5-large':
                current_chunk = []
                current_word_count = 0
                for sentence in lst:
                    sentence_word_count = len(sentence.split())
                    if current_word_count + sentence_word_count > n:
                        yield ' '.join(current_chunk)
                        current_chunk = [sentence]
                        current_word_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_word_count += sentence_word_count
                if current_chunk:
                    yield ' '.join(current_chunk)
            else:
                current_chunk = []
                current_token_count = 0
                for sentence in lst:
                    sentence_word_count = len(self.tokenizer(
                        sentence, padding=False, add_special_tokens=False, 
                        max_length=self.max_model_len, truncation=True)['input_ids'])
                    if current_token_count + sentence_word_count > n:
                        yield ' '.join(current_chunk)
                        current_chunk = [sentence]
                        current_token_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_token_count += sentence_word_count
                if current_chunk:
                    yield ' '.join(current_chunk)

        doc_sents = sent_tokenize_with_newlines(doc)
        doc_sents = doc_sents or ['']

        doc_chunks = [chunk.replace(" \n ", '\n').strip() for chunk in chunks(doc_sents, self.chunk_size)]
        doc_chunks = [chunk for chunk in doc_chunks if chunk != '']

        '''
        [chunk_1, chunk_2, chunk_3, chunk_4, ...]
        [claim]
        '''
        claim_repeat = [claim] * len(doc_chunks)
        
        output = self.inference(doc_chunks, claim_repeat)
        
        return output

    def inference(self, doc, claim):
        """
        inference a list of doc and claim

        Standard aggregation (max) over chunks of doc

        Note: We do not have any post-processing steps for 'claim'
        and directly check 'doc' against 'claim'. If there are multiple 
        sentences in 'claim'. Sentences are not splitted and are checked 
        as a single piece of text.
        
        If there are multiple sentences in 'claim', we suggest users to 
        split 'claim' into sentences beforehand and prepares data like 
        (doc, claim_1), (doc, claim_2), ... for a multi-sentence 'claim'.

        **We leave the user to decide how to aggregate the results from multiple sentences.**

        Note: AggreFact-CNN is the only dataset that contains three-sentence 
        summaries and have annotations on the whole summaries, so we do not 
        split the sentences in each 'claim' during prediciotn for simplicity. 
        Therefore, for this dataset, our result is based on treating the whole 
        summary as a single piece of text (one 'claim').

        In general, sentence-level prediciton performance is better than that on 
        the full-response-level.
        """

        if isinstance(doc, str) and isinstance(claim, str):
            doc = [doc]
            claim = [claim]
        
        batch_input, _, batch_org_chunks = self.batch_tokenize(doc, claim)

        label_probs_list = []
        used_chunks = []

        for mini_batch_input, batch_org_chunk in zip(batch_input, batch_org_chunks):

            mini_batch_input = {k: v.to(self.model.device) for k, v in mini_batch_input.items()}

            with torch.no_grad():

                decoder_input_ids = torch.zeros((mini_batch_input['input_ids'].size(0), 1), dtype=torch.long).to(self.model.device)
                outputs = self.model(input_ids=mini_batch_input['input_ids'], attention_mask=mini_batch_input['attention_mask'], decoder_input_ids=decoder_input_ids)
                logits = outputs.logits.squeeze(1)

                # 3 for no support and 209 for support
                label_logits = logits[:, torch.tensor([3, 209])].cpu()
                label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

                label_probs_list.append(label_probs)
                used_chunks.extend(batch_org_chunk)

        label_probs = torch.cat(label_probs_list)
        support_prob_per_chunk = label_probs[:, 1].cpu().numpy()
        max_support_prob = label_probs[:, 1].max().item()
        
        return {
            'max_support_prob': max_support_prob,
            'used_chunks': used_chunks,
            'support_prob_per_chunk': support_prob_per_chunk
        }

    def batch_tokenize(self, doc, claim):
        """
        input doc and claims are lists
        """
        assert isinstance(doc, list) and isinstance(claim, list)
        assert len(doc) == len(claim), "doc and claim should be in the same length."

        original_text = [self.tokenizer.eos_token.join([one_doc, one_claim]) for one_doc, one_claim in zip(doc, claim)]

        batch_input = []
        batch_concat_text = []
        batch_org_chunks = []
        for mini_batch in self.chunks(original_text, self.batch_size):
            model_inputs = self.tokenizer(
                ['predict: ' + text for text in mini_batch], 
                max_length=self.max_model_len, 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            ) 
            batch_input.append(model_inputs) 
            batch_concat_text.append(mini_batch)  
            batch_org_chunks.append([item[:item.find('</s>')] for item in mini_batch]) 

        return batch_input, batch_concat_text, batch_org_chunks

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def fact_check(self, doc, claim):

        outputs = self.inference_example_batch(doc, claim)
        return outputs['max_support_probs'], outputs['used_chunks'], outputs['support_prob_per_chunk']