import json
import os

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset


class Autodata:
    def __init__(self, data_folder="./data"):
        self.data_foloder = data_folder
        self.concat_dataset = self.concat_datasets(self.data_foloder)

    def concat_datasets(self, data_foloder):
        questions = []
        for file_name in os.listdir(data_foloder):
            if file_name.endswith(".json"):
                file_path = os.path.join(data_foloder, file_name)
                with open(file_path) as f:
                    question = json.load(f)["question"]
                questions.append(question)

        dataframe = pd.DataFrame(questions, columns=["question"])
        dataset = Dataset.from_pandas(dataframe)

        return dataset

    def load_instruction_dataset(self, dataset_id):
        koalpaca_data = load_dataset(dataset_id)
        data = koalpaca_data["train"]
        data = data.rename_column("instruction", "question")
        question = data["question"]
        return question

    def label_indexing(self, data, state):
        if state == 1:
            answer = 1
        else:
            answer = 0
        answer_list = [answer] * len(data)

        return Dataset.from_dict({"question": data, "target": answer_list})