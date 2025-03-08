from dataclasses import dataclass
import re
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from spacy.lang.en import STOP_WORDS
from tqdm import tqdm
from spacy.lang.en import English
import torch
import pandas as pd
import os

@dataclass
class DataTransformationConfig:
    sentence_chunk_size = 10
    embedding_model_name = "all-MiniLM-L12-v2"

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def convert_to_lowercase(self, text: str) -> str:
        new_text = text.lower()
        return new_text

    def remove_stopwords(self, text: str) -> str:
        new_text = []
        for word in text.split():
            if word not in STOP_WORDS:
                new_text.append(word)
        return " ".join(new_text)

    def remove_html_tags(self, text: str) -> str:
        new_text = re.sub(r"<.*?>", "", text)
        return new_text

    def remove_newlines(self, text: str) -> str:
        new_text = re.sub(r"\n+", " ", text)
        return new_text

    def remove_multiple_spaces(self, text: str) -> str:
        new_text = re.sub(r"\s+", " ", text)
        return new_text

    def remove_html_comments(self, text: str) -> str:
        new_text = re.sub(r"<!--.*?-->", "", text)
        return new_text

    def remove_unnecessary_text(self, text: str) -> str:
        new_text = text.replace("answer:","")
        new_text = new_text.replace("question","")
        new_text = new_text.replace(":","")
        return new_text

    def preprocess_text(self, text: str) -> str:
        text = self.convert_to_lowercase(text)
        text = self.remove_stopwords(text)
        text = self.remove_html_tags(text)
        text = self.remove_newlines(text)
        text = self.remove_multiple_spaces(text)
        text = self.remove_html_comments(text)
        text = self.remove_unnecessary_text(text)
        return text

    def preprocess_meta_data(self, pages_and_metadata: dict) -> dict:
        for page in tqdm(pages_and_metadata, total=len(pages_and_metadata)):
            page["formatted_text"] = self.preprocess_text(page["raw_text"])
        return pages_and_metadata

    def convert_paragraphs_to_sentences(self, pages_and_metadata: dict) -> dict:
        nlp = English()
        nlp.add_pipe("sentencizer")
        for page in tqdm(pages_and_metadata, total=len(pages_and_metadata)):
            sentences = nlp(page["formatted_text"]).sents
            sentences = list(set([str(sentence) for sentence in sentences if len(str(sentence).split())>10]))
            pages_and_metadata[page["page_number"]]["sentences"] = sentences
            pages_and_metadata[page["page_number"]]["number_of_sentences"] = len(sentences)
        return pages_and_metadata

    def convert_sentences_to_sentence_chunks(self, pages_and_metadata: dict) -> dict:
        for page in tqdm(pages_and_metadata, total=len(pages_and_metadata)):
            sentences = page["sentences"]
            sentence_chunk = [sentences[i: i+self.data_transformation_config.sentence_chunk_size] for i in range(0, len(sentences), self.data_transformation_config.sentence_chunk_size)]
            pages_and_metadata[page["page_number"]]["sentence_chunks"] = sentence_chunk
        return pages_and_metadata

    def convert_sentences_to_sentence_chunk_embeddings(self, pages_and_metadata):
        embedding_model = SentenceTransformer(self.data_transformation_config.embedding_model_name)
        for page in tqdm(pages_and_metadata, total=len(pages_and_metadata)):
            embeddings = list()
            for sentence in page["sentences"]:
                embedding = embedding_model.encode(
                    sentence,
                    batch_size=1024,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embedding = np.stack(embedding.tolist(), axis=0)
                embedding = torch.tensor(embedding)
                embedding = embedding.type(torch.float32)
                embeddings.append(embedding)
            sentence_embeddings = [np.array(embedding) for embedding in embeddings]
            pages_and_metadata[page["page_number"]]["embeddings"] = sentence_embeddings
        return pages_and_metadata

    def get_data_embeddings(self, pages_and_metadata: dict) -> list:
        pages_and_metadata_embeddings = list()
        for page in tqdm(pages_and_metadata, total=len(pages_and_metadata)):
            page_embeddings = list()
            for chunk_embedding in page["embeddings"]:
                if isinstance(chunk_embedding, torch.Tensor):
                    chunk_embedding = chunk_embedding.tolist()
                page_embeddings.append(chunk_embedding)
            pages_and_metadata_embeddings.append(page_embeddings)
        return pages_and_metadata_embeddings

    def convert_embeddings_to_same_dimensions(self, pages_and_metadata_embeddings: List) -> List:
        if pages_and_metadata_embeddings:
            embedding_dim = len(pages_and_metadata_embeddings[0][0])
            pages_and_metadata_embeddings = [
                [np.pad(chunk, (0, max(0, embedding_dim - len(chunk))), mode='constant')[:embedding_dim]
                 for chunk in page]
                for page in pages_and_metadata_embeddings
            ]
        return pages_and_metadata_embeddings
        return pages_and_metadata_embeddings

    def flatten(self, pages_and_metadata_embeddings, pages_and_metadata):
        flat_embeddings = [chunk for page in pages_and_metadata_embeddings for chunk in page]
        flat_data = [sentence for page in pages_and_metadata for sentence in page["sentences"]]
        return flat_data, flat_embeddings

    def save(self, flat_data, flat_embeddings):
        os.makedirs("artifacts", exist_ok=True)
        df = pd.DataFrame(flat_embeddings)
        df.to_csv("artifacts\\embeddings.csv", index=False)
        df = pd.DataFrame(flat_data)
        df.to_csv("artifacts\\data.csv", index=False)

    def main(self, pages_and_metadata):
        pages_and_metadata = self.preprocess_meta_data(pages_and_metadata)
        pages_and_metadata = self.convert_paragraphs_to_sentences(pages_and_metadata)
        pages_and_metadata = self.convert_sentences_to_sentence_chunks(pages_and_metadata)
        pages_and_metadata = self.convert_sentences_to_sentence_chunk_embeddings(pages_and_metadata)
        pages_and_metadata_embeddings = self.get_data_embeddings(pages_and_metadata)
        pages_and_metadata_embeddings = self.convert_embeddings_to_same_dimensions(pages_and_metadata_embeddings)
        flat_data, flat_embeddings = self.flatten(pages_and_metadata_embeddings, pages_and_metadata)
        self.save(flat_data, flat_embeddings)
        return pages_and_metadata