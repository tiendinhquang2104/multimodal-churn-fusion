# Multimodal Churn Fusion

Multimodal Customer Churn Prediction using Transaction, Text, and Image Data.

## Current milestone: customer feature table

The first runnable baseline stops at a feature table. **One row is one customer**:

```text
customer_id | transaction_000 ... transaction_127 | text_000 ... text_127 | image_000 ... image_127
```

There is no churn label, 0/1 prediction, or classifier output in this table. Those belong to the next stage after the observation and future prediction windows are defined.

The three branches are:

1. A randomly initialized **Transaction Transformer** reads up to 32 observed purchases per customer and emits one 128D vector.
2. Frozen **MiniLM** encodes each purchased product's name and description once. Purchased-product vectors are averaged per customer and projected to 128D.
3. Frozen **CLIP-ViT** encodes each available purchased-product image once. Those vectors are averaged per customer and projected to 128D.

The text and image 128D projections are seeded Gaussian matrices. The transaction Transformer and projections are **not trained**. The output is a reproducible pipeline and feature-shape milestone, **not evidence of churn prediction quality**. The projection matrices and transaction model state are saved beside the CSV.

## Data and alignment

The H&M archive contains `transactions_train.csv`, `articles.csv`, and `images/`; image files are grouped by the first three digits of `article_id`, and some products lack images. See the [official H&M dataset description](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).

Only purchases on or before the supplied `observation_end` enter the feature table. `observation_start` is optional. The cutoff in the example notebook is **for a technical demonstration only**; the thesis has not fixed observation or prediction-window lengths. Missing text or image data produces a zero vector for that modality. Repeated purchases contribute repeatedly to the customer mean.

The sample limit chooses the first encountered customers and scans the full transaction CSV to retain their complete observed histories. It is **not a research sampling scheme**. The script keeps only the most recent configured transactions per customer. The large ZIP is read in place and is not extracted into the repository.

## Repository structure

- `configs/`: baseline, dataset location, and experiment overrides.
- `src/multimodal_churn/data/`: H&M ZIP reader, sequence preparation, pretrained product encoders, and customer pooling.
- `src/multimodal_churn/models/`: transaction Transformer and later model components.
- `src/multimodal_churn/feature_pipeline.py`: table generation and artifact writing.
- `notebooks/00_setup_colab.ipynb`: Colab installation and Drive setup.
- `notebooks/05a_build_feature_table.ipynb`: first feature-table run.
- `scripts/build_feature_table.py`: command-line version.
- `data/` and `outputs/`: local directories excluded from version control.

## Installation and Colab run

Use Python 3.10 or newer and a GPU runtime in Colab:

```bash
pip install -r requirements.txt
pip install -e .
```

Run `notebooks/00_setup_colab.ipynb`, then `notebooks/05a_build_feature_table.ipynb`. The first feature run downloads the public MiniLM and CLIP model weights through their libraries. The H&M dataset is **not** downloaded by this project.

The H&M ZIP is in the [shared `02_hm_fashion` folder](https://drive.google.com/drive/folders/1xFkfs4Jcm_fH1dOZFX73PTYs0K7O-N97) (about 30.8 GB), under the shared [Project Master Thesis root](https://drive.google.com/drive/folders/1YHWbumTiVjEZ1evW-TV_ZctoUh3CABR8). A Drive URL is not a mounted filesystem path. In Colab, authorize the same account for the Drive mount and Drive API. If the archive is absent from the mounted My Drive, the setup notebook creates a shortcut to the shared project root in My Drive and remounts it. The ZIP is then read at `MyDrive/Project Master Thesis/Data/Datasets/02_hm_fashion/h-and-m-personalized-fashion-recommendations.zip` without copying it. If API authorization fails, add the shortcut manually with **Organize → Add shortcut → My Drive**. The notebooks also recognize a `Datasets` or `02_hm_fashion` shortcut. If your shortcut is elsewhere, set `DATASETS_ROOT_OVERRIDE` to the parent of `02_hm_fashion`. A failed `drive.mount()` must be resolved before reading the ZIP; reconnect the Colab runtime and reauthorize the correct account.

The equivalent command is:

```bash
python scripts/build_feature_table.py \
  --datasets-root "/content/drive/MyDrive/Project Master Thesis/Data/Datasets" \
  --observation-end 2020-08-31 \
  --max-customers 16 \
  --output-dir "/content/drive/MyDrive/Project Master Thesis/Data/features/hm_demo_2020-08-31"
```

`2020-08-31` and `16` are demo choices, not finalized research settings. The output directory receives `features.csv`, `manifest.json`, `projection_matrices.npz`, and `transaction_encoder.pt`.

## V1 temporal SSL feature table

The existing command remains the `baseline` mode. The new `temporal_ssl` mode uses **three modality-specific Temporal Transformers** with separate weights and a shared implementation. Frozen MiniLM and CLIP are product backbones; the temporal branches learn customer histories. Transaction events use normalized price, a two-category channel embedding, and log-scaled inter-purchase days. Each branch produces a raw 128D customer vector. The SSL projection heads are used only for NT-Xent training and are not exported.

```bash
python scripts/build_feature_table.py \
  --mode temporal_ssl \
  --datasets-root "/content/drive/MyDrive/Project Master Thesis/Data/Datasets" \
  --observation-end 2020-08-31 \
  --output-dir "/content/hm_temporal_ssl" \
  --cache-dir "/content/drive/MyDrive/Project Master Thesis/Data/features/hm_temporal_ssl_2020-08-31/product_cache" \
  --resume-dir "/content/drive/MyDrive/Project Master Thesis/Data/features/hm_temporal_ssl_2020-08-31"
```

Omitting `--max-customers` processes all customers in the observation window. The first full run extracts frozen product embeddings and trains three separate branches, so use a GPU and enough local disk space for the SQLite staging file, history arrays, checkpoints, and Parquet output. The Colab notebook writes the product cache to Drive and copies a verified `latest_*_ssl.pt` checkpoint after every epoch, along with each improved `best_*_ssl.pt` and metrics. After a runtime reset, rerun setup and the full-run cell with the same cutoff, configuration, and Drive output folder. Histories are rebuilt from the ZIP, then completed epochs and branches are skipped. A run with changed inputs/configuration is rejected rather than silently loading incompatible weights. Direct CLI runs can use `--cache-dir` and `--resume-dir` for persistent cache/checkpoints, but must separately copy the finished feature table and other artifacts; `--csv` is an optional compatibility export. A small `--max-customers` run may lack enough SSL validation customers; it is not a research cohort.

The canonical output is `features_temporal_ssl.parquet`: `customer_id` plus 384 float32 columns, with no label. `features_temporal_ssl_metadata.parquet` records availability and original/used history lengths; `manifest_temporal_ssl.json` records the cutoff, backbone revisions, split, normalization, and checkpoint identities. `intermediate/` retains sequence arrays and `checkpoints/` retains the best validation checkpoint for each branch. Missing modalities receive zero vectors and an availability flag in the sidecar.

Run `notebooks/05b_temporal_ssl_colab.ipynb` after setup to execute the synthetic architecture and end-to-end tests on Colab. Its full H&M cell is opt-in. Epoch checkpoints are stored in `MyDrive/Project Master Thesis/Data/features/hm_temporal_ssl_<observation_end>/checkpoints`; product embeddings are cached in its `product_cache` folder. The cell copies all remaining artifacts after a successful full run and verifies SHA-256 checksums. The Drive copy is marked complete only after every file has been checked. If copying is interrupted while `/content` still exists, rerun the final notebook cell to retry it. An interrupted epoch is repeated on resume; staging and history construction also run again because their working files remain in `/content`.

Current runs are marked `pretraining_scope=all_customers_debug`. For a final inductive churn benchmark, split customers before SSL, fit normalization and pretrain only on training customers, then freeze the checkpoints to encode validation and test customers. BG/NBD label construction is a separate later stage; V1 neither defines its threshold nor joins labels. **V1 constructs self-supervised customer-level multimodal representations and a reusable 384-dimensional feature table. It does not yet constitute evidence of churn-prediction performance.**

## Future datasets and experiment roadmap

Dataset location is configured using `configs/datasets/<name>.yaml`, relative to a runtime `DATASETS_ROOT`. New raw schemas will need their own small data adapter while reusing the customer pooling, projection, and table-assembly functions.

The planned ablations are transaction only, transaction + text, transaction + image, and all three modalities with concatenation. Text-only and image-only configs remain optional. Churn-label construction, leakage-safe supervised splits, classifier training, and predictive evaluation are future stages. Concat is a baseline, not the thesis novelty.

## Data policy

Do not commit raw or processed datasets, embeddings, feature tables, checkpoints, secrets, or notebook caches. See `data/README.md`.
