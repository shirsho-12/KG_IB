from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import pandas as pd


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


class MuSiQueDataset(BaseDataset):
    def __init__(self, data, partition: str = "validation"):
        super().__init__(data=data, partition=partition)

    def __getitem__(self, idx):
        sample = self.data[idx]
        id_ = sample["id"]
        question = sample["question"]
        answer = [sample["answer"]]
        if len(sample["answer_aliases"]) > 0:
            answer.extend(sample["answer_aliases"])
        paragraphs = sample["paragraphs"]
        samples = []
        for para in paragraphs:
            title = para["title"]
            para_sentences = para["paragraph_text"]
            text = "Title: " + title + "\nParagraph: " + para_sentences + "\n"
            samples.append(text)
        return {"id": id_, "question": question, "answer": answer, "text": samples}


class TwoWikiMultiHopQADataSet(BaseDataset):
    def __init__(self, data_path, partition: str = "dev"):
        self.data = json.loads(
            Path(data_path / (partition + ".json")).read_text(encoding="utf-8")
        )

    def __getitem__(self, idx):
        sample = self.data[idx]
        id_ = sample["_id"]
        question = sample["question"]
        answer = sample["answer"]
        paragraphs = sample["context"]
        samples = []
        for passage in paragraphs:
            title = passage[0]
            para_sentences = passage[1]
            text = "Title: " + title + "\nParagraph: " + " ".join(para_sentences) + "\n"
            samples.append(text)

        return {"id": id_, "question": question, "answer": answer, "text": samples}


class SubQADataSet(BaseDataset):
    def __init__(self, data_path, question_files):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = pd.read_json(f)
        self.question_df = pd.DataFrame()
        for question_file in question_files:
            with open(question_file, "r", encoding="utf-8") as f:
                questions = pd.read_json(f)
                questions.set_index("_id", inplace=True)
                questions = questions[["question", "answer"]]
                # remove last 5 characters from index
                questions.index = questions.index.str[:-5]
                self.question_df = self.question_df.join(
                    questions, how="outer", rsuffix=f"_{len(self.question_df.columns)}"
                )
        self.data.set_index("_id", inplace=True)

    def __getitem__(self, index: int):
        sample = self.data.iloc[index].to_dict()
        id_ = self.data.index[index]
        questions, answers = {}, {}
        question = sample["question"]
        answer = sample["answer"]
        paragraphs = sample["context"]
        samples = []
        for passage in paragraphs:
            title = passage[0]
            para_sentences = passage[1]
            text = "Title: " + title + "\nParagraph: " + " ".join(para_sentences) + "\n"
            samples.append(text)

        questions["main"] = question
        answers["main"] = answer
        question_entry = self.question_df.loc[id_]
        questions["sub_1"] = question_entry["question"]
        answers["sub_1"] = question_entry["answer"]
        questions["sub_2"] = question_entry["question_2"]
        answers["sub_2"] = question_entry["answer_2"]

        return {"id": id_, "questions": questions, "answers": answers, "text": samples}


def collate_fn(batch):
    ids = [item["id"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    texts = [item["text"] for item in batch]
    return {"ids": ids, "questions": questions, "answers": answers, "texts": texts}


def subqa_collate_fn(batch):
    ids = [item["id"] for item in batch]
    questions = [item["questions"] for item in batch]
    answers = [item["answers"] for item in batch]
    texts = [item["text"] for item in batch]
    return {"ids": ids, "questions": questions, "answers": answers, "texts": texts}


def get_hotpot_dataloader(data, partition="validation", batch_size=8, shuffle=False):
    dataset = HotpotDataset(data=data, partition=partition)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


def get_musique_dataloader(data, partition="validation", batch_size=8, shuffle=False):
    dataset = MuSiQueDataset(data=data, partition=partition)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


def get_2wikimultihopqa_dataloader(
    data_path, partition="dev", batch_size=8, shuffle=False
):
    dataset = TwoWikiMultiHopQADataSet(data_path=data_path, partition=partition)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


def get_subqa_dataloader(data_path, question_files, batch_size=8, shuffle=False):
    dataset = SubQADataSet(data_path=data_path, question_files=question_files)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=subqa_collate_fn
    )
    return dataloader
