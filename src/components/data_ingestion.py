import fitz
import os
from dataclasses import dataclass
from typing import List
from glob import glob
from tqdm import tqdm


@dataclass
class DataIngestionConfig:
    pdf_paths: str = os.path.join(os.path.dirname(__file__), "data", "*.pdf")
    skip_pages: int = 15


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.pdf_paths = glob(self.data_ingestion_config.pdf_paths)

    def open_documents(self, pdf_paths: List[str]) -> List[fitz.Document]:
        documents = []
        for path in tqdm(pdf_paths):
            doc = fitz.open(path)
            documents.append(doc)
        return documents

    def get_text_from_documents(self, documents: List[fitz.Document]) -> dict:
        pages = dict()
        for doc in tqdm(documents, total=len(documents)):
            for page_number, page in enumerate(doc):
                if page_number >= self.data_ingestion_config.skip_pages:
                    page_number = len(pages)
                    pages[page_number] = page.get_text()
        return pages

    def get_meta_data_from_pages(self, pages: dict) -> dict:
        pages_and_metadata = list()
        for page_number, page in tqdm(pages.items(), total=len(pages)):
            metadata = dict()
            metadata["page_number"] = page_number
            metadata["raw_text"] = page
            metadata["number_of_characters"] = len(page)
            metadata["number_of_tokens"] = len(page)/4
            metadata["number_of_words"] = len(page.split())
            pages_and_metadata.append(metadata)
        return pages_and_metadata

    def main(self):
        documents = self.open_documents(self.pdf_paths)
        pages = self.get_text_from_documents(documents)
        pages_and_metadata = self.get_meta_data_from_pages(pages)
        return pages_and_metadata