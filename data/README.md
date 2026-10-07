# Local data layout

- `raw/`: original downloaded datasets, kept unchanged.
- `processed/`: cleaned and transformed data.
- `embeddings/`: precomputed product text and image embeddings.
- `splits/`: train, validation, and test split files.

The H&M dataset is not included. These directories are ignored by Git. On Google Colab, the current H&M ZIP is in the mounted Drive path `MyDrive/Project Master Thesis/Data/Datasets/02_hm_fashion/h-and-m-personalized-fashion-recommendations.zip`. Set the datasets root in `notebooks/00_setup_colab.ipynb`; `configs/datasets/hm.yaml` contains only the relative folder and archive name. Do not copy or extract the archive into the repository.

For another dataset, add a dataset YAML with its own relative folder and optional archive. For frozen pretrained encoders, precompute text and image embeddings instead of recomputing them every epoch.

The first H&M feature-table run writes `features.csv`, `manifest.json`, and encoder/projection state under the notebook's `ARTIFACTS_ROOT/features/` directory on Drive. It reads the ZIP in place and does not create churn labels.
