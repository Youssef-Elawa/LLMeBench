import json

from llmebench.datasets import PaLMEvalDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import MultiNativQATask


def metadata():
    return {
        "author": "UBC-NLP / Adapted by QCRI",
        "model": "OpenAIModel",
        "description": "Evaluation on PaLM dataset containing MSA and dialect instructions across 22 Arab countries.",
        "scores": {},
    }


def config():
    return {
        "dataset": PaLMEvalDataset,
        "task": MultiNativQATask,
        "model": OpenAIModel,
        "general_args": {"test_split": "default"},
    }


def prompt(input_sample):
    # Define the question prompt
    question_prompt = f"""
        Please use your expertise to answer the following Arabic question. Answer in Arabic. Please provide Answer only. No additional text. 

        Question: {input_sample['question']}

        """

    # Define the assistant prompt
    assistant_prompt = """
    You are an Arabic AI assistant specialized in providing detailed and accurate answers across various fields. Your task is to deliver clear, concise, and relevant information. 
    """
    return [
        {"role": "user", "content": question_prompt},
        {"role": "assistant", "content": assistant_prompt},
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].strip()
    return content
