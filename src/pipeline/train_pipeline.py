from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_transformation = DataTransformation()
    pages_and_metadata = data_ingestion.main()
    pages_and_metadata = data_transformation.main(pages_and_metadata)