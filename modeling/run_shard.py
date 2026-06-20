"""
Run one shard of an experiment run-list on the GPU visible to this process.

Launched (once per GPU) by the dual-GPU notebook with CUDA_VISIBLE_DEVICES set, e.g.:
    CUDA_VISIBLE_DEVICES=0 python run_shard.py --runlist runlist.json --shard 0 --nshards 2 \
        --data-dir nse_data/dhan --out results/shard0.csv --ckpt-dir checkpoints/featabl \
        --area feature_ablation --epochs 20

The run-list is a JSON list of specs {model, symbol, horizon, feature_set, label_scheme, seed}.
This process runs specs[shard::nshards], appending rows to --out (its own file, so the two shards
never write the same CSV) and pushing checkpoints to S3 per run if AWS creds are present.
"""

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # make modeling/ importable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runlist", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--area", default="feature_ablation")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    import runner
    from nbenv import s3_client
    from train import DEVICE

    specs = json.loads(pathlib.Path(args.runlist).read_text())
    mine = specs[args.shard :: args.nshards]
    print(
        f"[shard {args.shard}/{args.nshards}] device={DEVICE} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} | {len(mine)} runs"
    )
    s3 = s3_client()
    for i, spec in enumerate(mine, 1):
        print(f"\n[shard {args.shard}] run {i}/{len(mine)}: {spec}")
        try:
            runner.run_one(
                spec,
                data_dir=args.data_dir,
                out_csv=args.out,
                ckpt_dir=args.ckpt_dir,
                area=args.area,
                s3=s3,
                epochs=args.epochs,
            )
        except Exception as e:  # keep the shard alive if one run fails
            print(f"[shard {args.shard}] run FAILED: {spec} -> {e!r}")
    print(f"[shard {args.shard}] done.")


if __name__ == "__main__":
    main()
