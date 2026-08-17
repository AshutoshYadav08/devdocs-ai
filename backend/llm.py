import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLM:

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.model = "gpt-4.1-mini"

    def generate(self, prompt):

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text