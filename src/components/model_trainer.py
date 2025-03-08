import pandas as pd
from dataclasses import dataclass
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import re


@dataclass
class SemanticSearchConfig:
    flat_data_path: str = "artifacts/data.csv"
    flat_embeddings_path: str = "artifacts/embeddings.csv"
    embedding_model_name = "all-MiniLM-L12-v2"
    tok_k = 15

class SemanticSearch:
    def __init__(self, query):
        self.semantic_search_config = SemanticSearchConfig()
        self.query = query

    def load(self):
        flat_embeddings = pd.read_csv(SemanticSearchConfig.flat_embeddings_path).to_numpy()
        flat_data = pd.read_csv(SemanticSearchConfig.flat_data_path)["0"].tolist()
        return flat_embeddings, flat_data

    def conversion(self, flat_embeddings):
        pages_and_metadata_embeddings = np.array(flat_embeddings, dtype=np.float32)
        pages_and_metadata_embeddings = torch.tensor(pages_and_metadata_embeddings, dtype=torch.float32)
        return pages_and_metadata_embeddings

    def get_similarity_score(self, pages_and_metadata_embeddings):
        embedding_model = SentenceTransformer(self.semantic_search_config.embedding_model_name)
        query_embeddings = embedding_model.encode(self.query, convert_to_tensor=True).to("cpu")
        print("Query embeddings device: ",query_embeddings.device)
        print("Pages and metadata embeddings device: ", pages_and_metadata_embeddings.device)
        dot_scores = util.dot_score(query_embeddings, pages_and_metadata_embeddings)[0]
        return dot_scores

    def get_top_k_scores_and_indices(self, dot_scores):
        top_scores, top_indices = torch.topk(dot_scores, k=self.semantic_search_config.tok_k)
        return top_scores, top_indices

    def get_top_k_context(self, top_indices, flat_data):
        context = list()
        for index in top_indices:
            context.append(flat_data[index.item()])
        return context

    def main(self):
        flat_embeddings, flat_data = self.load()
        pages_and_metadata_embeddings = self.conversion(flat_embeddings)
        dot_scores = self.get_similarity_score(pages_and_metadata_embeddings)
        top_scores, top_indices = self.get_top_k_scores_and_indices(dot_scores)
        context = self.get_top_k_context(top_indices, flat_data)
        return context

@dataclass
class AugmentationConfig:
    llm_model_name = "google/gemma-2b-it"
    temperature = 0.3
    max_new_tokens = 512

class Augmentation(SemanticSearch):
    def __init__(self, query):
        super().__init__(query)
        self.query = query
        self.augmentation_config = AugmentationConfig()
        self.context = super().main()

    def load_llm_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.augmentation_config.llm_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
        return model

    def augmenting_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.augmentation_config.llm_model_name)
        context = "\n -".join(self.context)
        base_prompt = f'''Based on the following context items, please answer the query
        Context Items:
        {context}
        Query:
        {self.query}
        Answer:'''
        base_prompt = base_prompt.format(context=context, query=self.query)
        return base_prompt, tokenizer

    def apply_prompt_to_dialogue_template(self, base_prompt, tokenizer):
        dialogue_template = [{
            "role": "user",
            "content": base_prompt,
        }]
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return prompt

    def retrieve(self, tokenizer, llm_model, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt")
        outputs = llm_model.generate(
            **input_ids,
            temperature = self.augmentation_config.temperature,
            do_sample = True,
            max_new_tokens = self.augmentation_config.max_new_tokens
        )
        output_text = tokenizer.decode(outputs[0])
        return output_text

    def clean_output_text(self, output_text):
        idx = output_text.find("Answer")
        answer = output_text[idx + 7:]
        answer = answer.replace("**", "")
        answer = answer.replace("<start_of_turn>model", "")
        answer = re.sub("<.*?>", "", answer)
        return answer

    def main(self):
        llm_model = self.load_llm_model()
        base_prompt, tokenizer = self.augmenting_prompt()
        prompt = self.apply_prompt_to_dialogue_template(base_prompt, tokenizer)
        output_text = self.retrieve(tokenizer, llm_model, prompt)
        output_text = self.clean_output_text(output_text)
        return output_text