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

The H&M ZIP is in the [shared `02_hm_fashion` folder](https://drive.google.com/drive/folders/1xFkfs4Jcm_fH1dOZFX73PTYs0K7O-N97) (about 30.8 GB). A Drive URL is not a mounted filesystem path. In Colab, authorize the account that can open that folder. If the folder is only under **Shared with me**, use **Organize → Add shortcut → My Drive** in Google Drive. The notebooks look for the ZIP under `MyDrive/Project Master Thesis/Data/Datasets/02_hm_fashion`, `MyDrive/Datasets/02_hm_fashion`, and `MyDrive/02_hm_fashion`. If your shortcut is elsewhere, set `DATASETS_ROOT_OVERRIDE` to the parent of `02_hm_fashion` in each notebook. A failed `drive.mount()` must be resolved before reading the ZIP; reconnect the Colab runtime and reauthorize the correct account.

The equivalent command is:

```bash
python scripts/build_feature_table.py \
  --datasets-root "/content/drive/MyDrive/Project Master Thesis/Data/Datasets" \
  --observation-end 2020-08-31 \
  --max-customers 16 \
  --output-dir "/content/drive/MyDrive/multimodal-churn-fusion/features/hm_demo_2020-08-31"
```

`2020-08-31` and `16` are demo choices, not finalized research settings. The output directory receives `features.csv`, `manifest.json`, `projection_matrices.npz`, and `transaction_encoder.pt`.

## Future datasets and experiment roadmap

Dataset location is configured using `configs/datasets/<name>.yaml`, relative to a runtime `DATASETS_ROOT`. New raw schemas will need their own small data adapter while reusing the customer pooling, projection, and table-assembly functions.

The planned ablations are transaction only, transaction + text, transaction + image, and all three modalities with concatenation. Text-only and image-only configs remain optional. Churn-label construction, leakage-safe splits, encoder training, classifier training, and predictive evaluation are future stages. Concat is a baseline, not the thesis novelty.

## Data policy

Do not commit raw or processed datasets, embeddings, feature tables, checkpoints, secrets, or notebook caches. See `data/README.md`.
