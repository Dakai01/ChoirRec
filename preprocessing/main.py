"""One-click preprocessing entry point: raw logs to id-form artifacts."""

from __future__ import annotations

import argparse

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "preprocessing"

import numpy as np

from .config import PreprocessConfig
from .writers import (
    write_group_ids,
    write_rec_samples,
    write_vocab,
)
from .features import (
    build_user_features,
    build_vocabularies,
    load_exposure_samples,
    load_user_records,
)
from .RQKmeans import RQKMeans
from .llm import LLMService, select_device
from .representation import build_group_artifacts


def run(cfg: PreprocessConfig) -> None:
    np.random.seed(cfg.seed)

    print(f"[1/5] Loading user features from {cfg.users_file} "
          f"and exposure labels from {cfg.exposure_samples_file} ...")
    records = load_user_records(cfg.users_file)
    users = build_user_features(records)
    vocab = build_vocabularies(records)
    samples = load_exposure_samples(cfg.exposure_samples_file, records)
    print(f"  users={len(users)}  samples={len(samples)}  "
          f"low_activity={sum(u.is_low_activity for u in users)}")

    print(f"[2/5] Synthesising LLM profiles & embeddings (device={select_device()}) ...")
    llm = LLMService(cfg.llm)
    profiles = [llm.synthesize_profile(u) for u in users]
    embeddings = llm.embed_profiles(profiles)
    print(f"  embeddings shape={embeddings.shape}")

    print("[3/5] RQ-KMeans hierarchical grouping ...")
    rq = RQKMeans(cfg.grouping)
    rq_result = rq.fit(embeddings)
    print(f"  codes shape={rq_result.codes.shape}")

    print("[4/5] Building group-aware representations ...")
    artifacts = build_group_artifacts(users, rq_result.codes, vocab, cfg.seq_max_len)
    print(f"  leaf groups={artifacts.num_leaf_groups}")

    print("[5/5] Writing artifacts to data/ ...")
    write_vocab(vocab, cfg.vocab_file)
    write_rec_samples(samples, records, vocab, artifacts, cfg.rec_samples_file)
    write_group_ids(records, artifacts, cfg.group_ids_file)
    print(f"  wrote {cfg.vocab_file}")
    print(f"  wrote {cfg.rec_samples_file}")
    print(f"  wrote {cfg.group_ids_file}")
    print("[done] preprocessing complete.")


def main():
    parser = argparse.ArgumentParser(description="Run ChoirRec preprocessing.")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override path to users.tsv.")
    parser.add_argument("--chat-model", type=str, default=None)
    parser.add_argument("--embed-model", type=str, default=None)
    args = parser.parse_args()

    cfg = PreprocessConfig()
    if args.data_path is not None:
        cfg.users_file = args.data_path
    if args.chat_model is not None:
        cfg.llm.chat_model = args.chat_model
    if args.embed_model is not None:
        cfg.llm.embed_model = args.embed_model

    run(cfg)


if __name__ == "__main__":
    main()
