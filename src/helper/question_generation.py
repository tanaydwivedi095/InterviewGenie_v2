import random
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class GemmaQuestionGeneratorConfig:
    llm_model_name: str = "google/gemma-2b-it"
    number_of_questions: int = 5
    question_types = {
        0: "definition",
        1: "formula",
        2: "scenario",
        3: "case-study",
        4: "algorithmic theory",
        5: "mathematics",
        6: "probability and statistics"
    }
    temperature = 1
    max_new_tokens = 2048

class GemmaQuestionGenerator:
    def __init__(self):
        self.gemma_question_generator_config = GemmaQuestionGeneratorConfig()

    def load_llm_model(self):
        llm_model = AutoModelForCausalLM.from_pretrained(
            self.gemma_question_generator_config.llm_model_name,
            torch_dtype=torch.float16
        )
        return llm_model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.gemma_question_generator_config.llm_model_name,
        )
        return tokenizer

    def generate_prompt(self, tokenizer):
        question_types = list()
        for i in range(self.gemma_question_generator_config.number_of_questions):
            random_idx = random.randint(0, len(self.gemma_question_generator_config.question_types)-1)
            question_types.append(self.gemma_question_generator_config.question_types[random_idx])
        question_types = "\n -".join(question_types)
        base_prompt = f"""
                Generate {self.gemma_question_generator_config.number_of_questions} questions for an Interview for the position of Machine Learning Engineer.
                Each question should be of type as mentioned below:
                {question_types}
                The output provided should be in format as given below:
                Question 1: Question Type 1 ....
                Question 2: Question Type 2 ....
                Question 3: Question Type 3 .....
                and so on.
                No extra text should be generated as answer.
                The case study questions should be well defined.
                The questions should only be of the specified type.
                """
        base_prompt = base_prompt.format(question_types=question_types, number_of_questions=self.gemma_question_generator_config.number_of_questions)
        dialogue_template = [{
            "role":"user",
            "message": base_prompt,
        }]
        prompt = tokenizer.apply_chat_template(
            conversation=dialogue_template,
            tokenize=False,
            add_generation_prompt=True
        )
        return base_prompt

    def generate_questions(self, llm_model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt")
        outputs = llm_model.generate(
            **input_ids,
            temperature = self.gemma_question_generator_config.temperature,
            do_sample = True,
            max_new_tokens = self.gemma_question_generator_config.max_new_tokens
        )
        output_text = tokenizer.batch_decode(outputs)[0]
        return output_text

    def post_process_questions(self, questions):
        questions = questions[questions.find("\n\n**") + 4:].replace("**", "").replace("\n", "")
        questions = questions.split("Question")
        questions = [question for question in questions if question]
        questions[-1] = questions[-1][:-5]
        return questions

    def main(self):
        llm_model = self.load_llm_model()
        tokenizer = self.load_tokenizer()
        prompt = self.generate_prompt(tokenizer)
        output_text = self.generate_questions(llm_model, tokenizer, prompt)
        questions = self.post_process_questions(output_text)
        return questions

@dataclass
class OpenAIQuestionGeneratorConfig:
    number_of_questions: int = 5
    question_type = {
        0: "definition",
        1: "formula",
        2: "scenario",
        3: "case-study",
        4: "algorithmic theory",
        5: "mathematics",
        6: "probability and statistics",
        7: "real life application",
    }

class OpenAIQuestionGenerator:
    def __init__(self, secret_key, difficulty):
        self.secret_key = secret_key
        self.openai_question_generator_config = OpenAIQuestionGeneratorConfig()
        self.difficulty = difficulty

    def load_llm_model(self):
        llm_model = OpenAI(api_key=self.secret_key)
        return llm_model

    def generate_prompt(self):
        question_types = list()
        for i in range(self.openai_question_generator_config.number_of_questions):
            random_idx = random.randint(0, len(self.openai_question_generator_config.question_type)-1)
            question_types.append(f"{self.openai_question_generator_config.question_type[random_idx]}")
        question_types = "\n -".join(question_types)
        base_prompt = f"""
                Generate {self.openai_question_generator_config.number_of_questions} questions for an Interview for the position of Machine Learning Engineer.
                Each question should be of type as mentioned below:
                {question_types}
                The output provided should be in format as given below:
                Question 1: Question Type 1 ....
                Question 2: Question Type 2 ....
                Question 3: Question Type 3 .....
                and so on.
                No extra text should be generated as answer.
                The case study questions should be well defined.
                The questions should only be of the specified type.
                Do not mention the type of question in the generated question.
                Make the questions hard on a scale of {self.difficulty} out of 10.
                """
        base_prompt = base_prompt.format(question_types=question_types,
                                         number_of_questions=self.openai_question_generator_config.number_of_questions,
                                         difficulty=self.difficulty)
        print("PROMPT: ",base_prompt)
        dialogue_template = [{
            "role": "user",
            "content": base_prompt
        }]
        return dialogue_template

    def generate_questions(self, llm_model, dialogue_template):
        questions = llm_model.chat.completions.create(
            model = "gpt-4o-mini",
            store = False,
            messages = dialogue_template
        )
        return questions

    def post_process_questions(self, questions):
        questions = questions.choices[0].message.content.split("Question")
        questions = [f"{question.strip().replace('\n', "")[3:]}" for question in questions if question]
        return questions

    def main(self):
        llm_model = self.load_llm_model()
        dialogue_template = self.generate_prompt()
        questions = self.generate_questions(llm_model, dialogue_template)
        questions = self.post_process_questions(questions)
        return questions