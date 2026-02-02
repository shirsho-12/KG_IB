import asyncio
from collections import defaultdict
from pathlib import Path
import pickle
from tqdm import tqdm

from src.clusterer import OnlineRelationClusterer
from src.kg import NXKnowledgeGraph
from src.pragma import PragmaticEquivalenceLearner
from src.redundancy_filter import RedundancyFilter
from src.utils import get_agents, process_tasks_asynchronously


class Pipeline:
    def __init__(self):
        self.extractor, self.embedder, self.typer = get_agents()
        self.clusterer = OnlineRelationClusterer()
        self.pragma_learner = PragmaticEquivalenceLearner()
        self.output_memo = {}

    def evaluate_dataset(
        self,
        dataloader,
        max_samples=None,
        save_every=100,
        concurrency=5,
        start_index=0,
        save_path_prefix="output/",
    ):
        stage_1_all_results = {}
        stage_2_all_results = {}
        full_log_text = ""
        sample_count = 0
        for batch in tqdm(dataloader, desc="Evaluating dataset"):
            if max_samples is not None and sample_count >= max_samples:
                break
            sample_count += len(batch["texts"])
            if sample_count < start_index:
                continue
            stage_1_results = self._run_stage_1(batch, concurrency=concurrency)
            # stage_2_results, log_text = self._run_stage_2(stage_1_results)
            if save_every and sample_count % save_every == 0:
                self.save_results(
                    stage_1_results,
                    f"{save_path_prefix}/stage_1_results_{sample_count}.pkl",
                )
                # self.save_results(
                #     stage_2_results, f"{save_path_prefix}/kg_{sample_count}.pkl"
                # )
                # with open(f"{save_path_prefix}/log_{sample_count}.txt", "w") as f:
                #     f.write(full_log_text + log_text)
            # full_log_text += log_text
            stage_1_all_results.update(stage_1_results)
            # stage_2_all_results.update(stage_2_results)

        # with open(f"{save_path_prefix}/final_log.txt", "w") as f:
        #     f.write(full_log_text)

        return stage_1_all_results, stage_2_all_results

    def update_memo_with_saved(self, folder_path):
        p = Path(folder_path)
        for file in p.glob("stage_1_results_*.pkl"):
            with open(file, "rb") as f:
                data = pickle.load(f)
                for sample_id, samples in data.items():
                    for sample in samples:
                        for triple in sample["data"]:
                            h, r, t = triple
                            text = sample["sentence"]
                            self.output_memo[text] = sample

    def save_results(self, results, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(results, f)

    def _run_stage_1(self, batch, concurrency):
        results = {}
        tasks = []

        batch_size = len(batch["texts"])
        for i in range(batch_size):
            texts = batch["texts"][i]
            sample_id = batch["ids"][i]
            for text in texts:
                tasks.append((sample_id, text))

        async def worker(payload):
            sample_id, text = payload
            processed = await self.process_line_async(text)
            self.output_memo[text] = processed
            return sample_id, processed

        loop = asyncio.get_event_loop()
        processed_results = loop.run_until_complete(
            process_tasks_asynchronously(tasks, worker, concurrency, desc="Stage 1")
        )

        grouped = defaultdict(list)
        for sample_id, processed in processed_results:
            grouped[sample_id].append(processed)
        for sample_id, group in grouped.items():
            results[sample_id] = group

        return results

    def _run_stage_2(self, stage_1_data):
        log_text = ""
        graph_dct = {}
        for idx, data in stage_1_data.items():
            log_text += f"Processing sample {idx}\n"
            clusterer = OnlineRelationClusterer()
            for sample in data:
                for triple in sample["data"]:
                    clusterer.process_triple(triple)

            log_text += f"Formed {len(clusterer.clusters)} clusters.\n"

            learner = PragmaticEquivalenceLearner(mi_threshold=0.25, min_pairs=2)

            equiv_classes, inverse_map = learner(clusterer)
            log_text += f"Equivalence classes: {len(equiv_classes)}\n"
            log_text += f"Inverse map: {len(inverse_map)}\n"

            kg = NXKnowledgeGraph()
            redundancy_filter = RedundancyFilter(kg, equiv_classes, inverse_map)

            for h, r_surface, t, cid, sentence in clusterer.fact_list:
                added = redundancy_filter.add_if_novel(
                    h, cid, t, surface=r_surface, sentence=sentence
                )
                if not added:
                    log_text += f"Redundant fact skipped: ({h}, {r_surface}, {t})\n"
            log_text += f"Final KG has {len(kg.G.nodes())} nodes and {len(kg.G.edges())} edges.\n"
            graph_dct[idx] = kg

        return graph_dct, log_text

    async def process_line_async(self, text):
        if text in self.output_memo:
            return self.output_memo[text]
        triples = await self.extractor.extract_async(text)
        if triples is None or len(triples) == 0:
            # one more attempt to extract
            triples = await self.extractor.extract_async(text)

        s_dct = {"sentence": text, "triples": triples}
        filtered = []
        for triple in triples:
            if triple is None:
                continue
            h, r, t = triple
            if h is None or r is None or t is None:
                continue
            if "Paragraph" in h or "Paragraph" in t:
                continue
            if "Title:" in h or "Title:" in t:
                continue
            filtered.append((h, r, t))

        if not filtered:
            s_dct["data"] = []
            return s_dct

        async def handle_triple(triple):
            if triple is None:
                return None
            h, r, t = triple
            emb_task = asyncio.create_task(self.embedder.aembedding_fn(r, text))
            type_h_task = asyncio.create_task(self.typer.assign_type_async(h))
            type_t_task = asyncio.create_task(self.typer.assign_type_async(t))
            embedding = await emb_task
            type_h, type_t = await asyncio.gather(type_h_task, type_t_task)
            return {
                "head": h,
                "relation": r,
                "tail": t,
                "embedding": embedding,
                "type_pair": (type_h, type_t),
                "sentence": text,
            }

        triple_tasks = [handle_triple(triple) for triple in filtered]
        processed_triples = await asyncio.gather(*triple_tasks)
        s_dct["data"] = [pt for pt in processed_triples if pt is not None]
        self.output_memo[text] = s_dct
        return s_dct
