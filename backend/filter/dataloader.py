import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        target,
        state,
        text_columns,
        max_length=256,
        model_name="monologg/koelectra-small-v3-discriminator",
    ):
        self.state = state
        self.data = data
        self.target = target
        self.text_columns = text_columns
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.state == "test":
            self.inputs = self.preprocessing(self.data)
        else:
            self.inputs, self.targets = self.preprocessing(self.data, self.target)

    def __getitem__(self, idx):
        if self.state == "test":
            return {"input_ids": torch.tensor(self.inputs[idx], dtype=torch.long)}
        else:
            return {
                "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
                "labels": torch.tensor(self.targets[idx], dtype=torch.long),
            }

    def __len__(self):
        return len(self.inputs)

    def tokenizing(self, data_list: list[str]) -> list:
        """
        토크나이징
            Args :
                data_list (list[str]): 토크나이징할 데이터의 배열
            Return :
                data (list) : 학습할 문장 토큰 리스트
        """
        data = []
        for item in tqdm(
            data_list, desc="Tokenizing", total=len(data_list)
        ):
            text = item
            # text = [item for text_column in self.text_columns]
            outputs = self.tokenizer(
                text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            data.append(outputs["input_ids"])
        return data

    def preprocessing(self, data, target):
        inputs = self.tokenizing(data)
        if self.state == "test":
            return inputs
        else:
            return inputs, target