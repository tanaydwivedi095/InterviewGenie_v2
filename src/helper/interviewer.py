from src.helper.question_generation import GemmaQuestionGenerator
from src.helper.question_generation import OpenAIQuestionGenerator
from dataclasses import dataclass
from openai import OpenAI

class InterviewerConfig:
    def __init__(self, username, generator_type, api_key, difficulty):
        self.username: str = username
        self.generator_type: str = generator_type
        self.api_key: str = api_key
        self.difficulty: int = difficulty

class Interviewer(InterviewerConfig):
    def __init__(self, username, generator_type, api_key=None, difficulty=None):
        super().__init__(username, generator_type, api_key, difficulty)

    def generate_questions(self):
        if self.generator_type == "Paid":
            questions = OpenAIQuestionGenerator(self.api_key, self.difficulty).main()
            return questions
        else:
            questions = GemmaQuestionGenerator().main()
            return questions

    def get_a_question(self, questions):
        question = questions[0]
        if len(questions) > 1:
            questions = questions[1:]
        else:
            questions = []
        return questions, question

    def score_answer(self, query, user_answer):
        new_prompt = f"""
            I have a question and an answer, I need you to rate the answer on a scale of 0 to 10,
            0 being the worst score and 10 being the best score.
            The question was {query}.
            The answer is {user_answer}.
            Score it from a perspective of an Interviewer of Machine Learning Engineer job post.
            I need the answer in a specific format that is 'score'
            I need no extra character to be generated.
            Keep the scoring easy for the initial scores but when the score is above 7 the answer need to be well described.
            Give some feedback points as paragraphs regarding how to improve the answer. 
            Put a <br> tag between the score and the feedback.
            Generate the response as HTML tags.
            Put in no extra text.
        """
        dialogue_template = [{
            "role":"user",
            "content": new_prompt
        }]

        llm_model = OpenAI(api_key=self.api_key)
        score = llm_model.chat.completions.create(
            model="gpt-4o-mini",
            store=False,
            messages=dialogue_template
        )
        score = score.choices[0].message.content
        return score