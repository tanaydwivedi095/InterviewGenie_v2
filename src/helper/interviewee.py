from src.components.model_trainer import Augmentation

def get_answer(query):
    aug = Augmentation(query)
    return aug.main()