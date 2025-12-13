import argparse
import asyncio
import pickle
from pathlib import Path

from tqdm import tqdm

from src.clusterer import OnlineRelationClusterer
from src.kg import NXKnowledgeGraph
from src.pragma import PragmaticEquivalenceLearner
from src.redundancy_filter import RedundancyFilter
from src.utils import (
    get_agents,
    process_line_async,
    process_tasks_asynchronously,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dset",
    type=str,
    required=True,
    choices=["rebel", "example", "webnlg", "wiki-nre"],
    help="Name of dataset",
)
parser.add_argument(
    "--concurrency",
    type=int,
    default=5,
    help="Maximum number of concurrent Stage 1 requests",
)


async def main():
    args = parser.parse_args()
    dset_name = args.dset
    input_path = Path(f"data/{dset_name}.txt")
    dset = input_path.read_text().splitlines()

    tier_2_dir = Path(f"output/{dset_name}/tier_2")
    tier_2_dir.mkdir(parents=True, exist_ok=True)
    stage_one_file = tier_2_dir / "s_1_extracted.pkl"
    legacy_stage_one_file = Path(f"output/{dset_name}/s_1_extracted.pkl")

    extractor, embedder, typer = get_agents()

    if stage_one_file.exists():
        print(f"Stage 1 data already exists at {stage_one_file}, skipping extraction.")
        with open(stage_one_file, "rb") as f:
            s_1_data = pickle.load(f)
    elif legacy_stage_one_file.exists():
        print(
            f"Stage 1 data already exists at {legacy_stage_one_file}, skipping extraction."
        )
        with open(legacy_stage_one_file, "rb") as f:
            s_1_data = pickle.load(f)
    else:
        async def worker(text):
            return await process_line_async(text, extractor, embedder, typer)

        s_1_data = await process_tasks_asynchronously(
            dset, worker, args.concurrency, "Stage 1"
        )
        with open(stage_one_file, "wb") as f:
            pickle.dump(s_1_data, f)
        print(f"Saved stage 1 data to {stage_one_file}")

    clusterer = OnlineRelationClusterer()
    for sample in tqdm(s_1_data, desc="Clustering", total=len(s_1_data)):
        for triple in sample["data"]:
            clusterer.process_triple(triple)

    print(f"Formed {len(clusterer.clusters)} clusters.")

    learner = PragmaticEquivalenceLearner(mi_threshold=0.25, min_pairs=2)

    equiv_classes, inverse_map = learner(clusterer)
    print("Equivalence classes:", len(equiv_classes))
    print("Inverse map:", len(inverse_map))

    kg = NXKnowledgeGraph()
    redundancy_filter = RedundancyFilter(kg, equiv_classes, inverse_map)

    for h, r_surface, t, cid, sentence in clusterer.fact_list:
        added = redundancy_filter.add_if_novel(
            h, cid, t, surface=r_surface, sentence=sentence
        )
        if not added:
            print(["REDUNDANT", "ACCEPTED"][added], h, r_surface, t, "→ cluster", cid)
    print(f"Final KG has {len(kg.G.nodes())} nodes and {len(kg.G.edges())} edges.")

    kg.save(tier_2_dir / "final_kg.graphml")
    print(f"Saved KG to {tier_2_dir}/final_kg.graphml")


if __name__ == "__main__":
    asyncio.run(main())
