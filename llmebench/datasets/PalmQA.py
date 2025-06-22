import json

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class PaLMEvalDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(PaLMEvalDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {
            "data_id": "1",
            "input": {
                "question": "من الملك الذي كان يتولى الحكم في الأردن عندما تم بناء مسجد الحسين؟"
            },
            "label": "بني مسجد الحسين في عهد الملك عبد الله الثاني.",
        }

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": "Refer to PaLM eval paper",
            "link": "https://github.com/UBC-NLP/palm",
            "license": "",
            "splits": {"default": {"test": "test.jsonl"}},
            "task_type": TaskType.Other,
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                # Concatenate instruction and input
                instruction = obj.get("instruction") or ""
                input_text = obj.get("input") or ""

                full_prompt = f"{instruction.strip()} {input_text.strip()}".strip()

                # Use "output" instead of "ideal"
                output = obj.get("output")
                if output is None:
                    print(f"Missing output for ID {obj.get('id')}")
                    output = ""

                label = output

                data.append(
                    {
                        "data_id": obj.get("id"),
                        "input": {"question": full_prompt},
                        "label": label,
                    }
                )

        return data
