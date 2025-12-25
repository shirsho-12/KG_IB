from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


class BaseDataset(Dataset):
    def __init__(self, data_path=None, data=None, partition: str = "validation"):
        if data is None:
            raise ValueError("Either data_path or data must be provided.")
        elif data_path is not None:
            data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.data = data[partition]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class HotpotDataset(BaseDataset):
    def __init__(self, data, partition: str = "validation"):
        super().__init__(data=data, partition=partition)

    def __getitem__(self, idx):
        sample = self.data[idx]
        id_ = sample["id"]
        question = sample["question"]
        answer = sample["answer"]
        paragraphs = sample["context"]
        n = len(paragraphs["title"])
        samples = []
        for i in range(n):
            title = paragraphs["title"][i]
            para_sentences = paragraphs["sentences"][i]
            text = "Title: " + title + "\nParagraph: " + " ".join(para_sentences) + "\n"
            samples.append(text)

        return {"id": id_, "question": question, "answer": answer, "text": samples}


def collate_fn(batch):
    ids = [item["id"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    texts = [item["text"] for item in batch]
    return {"ids": ids, "questions": questions, "answers": answers, "texts": texts}


def get_hotpot_dataloader(
    data=None, partition="validation", batch_size=8, shuffle=False
):
    dataset = HotpotDataset(data=data, partition=partition)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader
