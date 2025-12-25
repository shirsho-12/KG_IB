import argparse
from pathlib import Path

from datasets import load_dataset
import time
from src.dataloader import get_hotpot_dataloader
from src.pipeline import Pipeline
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples",
    type=int,
    default=9,
    help="Number of validation samples to process",
)
parser.add_argument(
    "--concurrency",
    type=int,
    default=3,
    help="Maximum number of concurrent Stage 1 requests",
)

parser.add_argument(
    "--save_every",
    type=int,
    default=8,
    help="Save results every N samples",
)


def main():
    args = parser.parse_args()
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", cache_dir="data/")
    # dset = ds["validation"]
    # print(len(dset))
    today = time.strftime("%d-%m-%Y")

    dset_name = "hotpotqa"
    output_path = Path(f"output/{dset_name}/{today}")
    output_file_path = output_path / "s_1_extracted.pkl"
    output_kg_file_path = output_path / "kg.pkl"
    output_path.mkdir(parents=True, exist_ok=True)

    dataloader = get_hotpot_dataloader(
        data=ds, partition="validation", batch_size=args.save_every, shuffle=False
    )
    pipeline = Pipeline()
    stage_1_results, stage_2_results = pipeline.evaluate_dataset(
        dataloader,
        max_samples=args.samples,
        save_every=args.save_every,
        concurrency=args.concurrency,
        save_path_prefix=f"output/{dset_name}/{today}/",
    )
    print(f"Stage 1 and Stage 2 results saved to {output_path}")
    with open(output_file_path, "wb") as f:
        pickle.dump(stage_1_results, f)
    with open(output_kg_file_path, "wb") as f:
        pickle.dump(stage_2_results, f)
    print(f"Stage 1 results saved to {output_file_path}")


if __name__ == "__main__":
    main()
