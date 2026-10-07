"""Build a customer-level H&M feature table without churn labels."""

import argparse
import logging
from pathlib import Path

from multimodal_churn.feature_pipeline import build_hm_feature_table
from multimodal_churn.utils.config import load_config, resolve_dataset_paths


def main() -> None:
    """Resolve the configured archive and export the feature table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/datasets/hm.yaml"))
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--observation-end", required=True, help="Inclusive YYYY-MM-DD cutoff.")
    parser.add_argument("--observation-start", default=None)
    parser.add_argument("--max-customers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "temporal_ssl"), default="baseline")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Optional shared frozen-product embedding cache for temporal_ssl.")
    parser.add_argument("--resume-dir", type=Path, default=None,
                        help="Persistent temporal_ssl directory for verified epoch checkpoints and resume.")
    parser.add_argument("--csv", action="store_true",
                        help="Also export a compatibility CSV in temporal_ssl mode.")
    args = parser.parse_args()
    config = load_config(args.baseline, dataset_path=args.dataset_config)
    paths = resolve_dataset_paths(config, args.datasets_root)
    if "archive" not in paths:
        parser.error("Selected dataset config must define dataset.archive.")
    if args.mode == "baseline":
        table = build_hm_feature_table(
            config,
            archive_path=paths["archive"],
            observation_start=args.observation_start,
            observation_end=args.observation_end,
            max_customers=args.max_customers,
            output_dir=args.output_dir,
        )
        print(f"Saved {len(table)} customer rows and {table.shape[1] - 1} features to "
              f"{args.output_dir / 'features.csv'}")
    else:
        from multimodal_churn.temporal_ssl_pipeline import build_temporal_ssl_feature_table

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        path = build_temporal_ssl_feature_table(
            config, paths["archive"], args.observation_end, args.output_dir,
            observation_start=args.observation_start,
            max_customers=args.max_customers, cache_dir=args.cache_dir,
            csv_compat=args.csv, resume_dir=args.resume_dir,
        )
        print(f"Saved temporal SSL customer features to {path}")


if __name__ == "__main__":
    main()
