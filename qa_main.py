import argparse
from pathlib import Path

from datasets import load_dataset
import time
from src.dataloader import (
    get_hotpot_dataloader,
    get_musique_dataloader,
    get_2wikimultihopqa_dataloader,
    get_subqa_dataloader,
    get_moby_dataloader,
)
from src.pipeline import Pipeline
import pickle

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="moby",
    help="Dataset to use: hotpotqa, musique, 2wikimultihopqa, subqa, or moby",
)

parser.add_argument(
    "--samples",
    type=int,
    default=-1,
    help="Number of validation samples to process",
)
parser.add_argument(
    "--concurrency",
    type=int,
    default=50,
    help="Maximum number of concurrent Stage 1 requests",
)

parser.add_argument(
    "--save_every",
    type=int,
    default=100,
    help="Save results every N samples",
)
parser.add_argument(
    "--start_index",
    type=int,
    default=100,
    help="Start from sample index",
)


def main():
    args = parser.parse_args()
    if args.dataset == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", cache_dir="data/")
        dataloader = get_hotpot_dataloader(
            data=ds, partition="validation", batch_size=args.save_every, shuffle=False
        )
    elif args.dataset == "musique":
        ds = load_dataset("dgslibisey/MuSiQue", cache_dir="data/")
        dataloader = get_musique_dataloader(
            data=ds, partition="validation", batch_size=args.save_every, shuffle=False
        )
    elif args.dataset == "2wikimultihopqa":
        data_path = Path("data/2wikimultihopqa/")
        dataloader = get_2wikimultihopqa_dataloader(
            data_path=data_path,
            partition="dev",
            batch_size=args.save_every,
            shuffle=False,
        )
    elif args.dataset == "subqa":
        data_path = Path("data/subqa/")
        dataloader = get_subqa_dataloader(
            data_path=data_path / "dev_ori.json",
            question_files=[
                data_path / "dev_sub1.json",
                data_path / "dev_sub2.json",
            ],
            batch_size=args.save_every,
            shuffle=False,
        )
    elif args.dataset == "moby":
        text_path = Path("data/moby_dick.txt")
        dataloader = get_moby_dataloader(
            text_path=text_path,
            batch_size=args.save_every,
            shuffle=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    today = time.strftime("%d-%m-%Y")

    dset_name = args.dataset
    output_path = Path(f"output/{dset_name}/{today}")
    output_file_path = output_path / "s_1_extracted.pkl"
    output_kg_file_path = output_path / "kg.pkl"
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline()
    stage_1_results, stage_2_results = pipeline.evaluate_dataset(
        dataloader,
        max_samples=None if args.samples == -1 else args.samples,
        save_every=args.save_every,
        concurrency=args.concurrency,
        start_index=args.start_index,
        save_path_prefix=f"output/{dset_name}/{today}/",
    )
    print(f"Stage 1 and Stage 2 results saved to {output_path}")
    with open(output_file_path, "wb") as f:
        pickle.dump(stage_1_results, f)
    # with open(output_kg_file_path, "wb") as f:
    #     pickle.dump(stage_2_results, f)
    print(f"Stage 1 results saved to {output_file_path}")


if __name__ == "__main__":
    main()
