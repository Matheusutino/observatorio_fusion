# Researcher-Topic Fusion

This repository implements an experimental pipeline for evaluating fusion operators
in the context of predicting the alignment between a researcher and a strategic
theme defined by the Brazilian National Postgraduate System (SNPG).

Postgraduate education in Brazil is a major pillar for national development, and the
CAPES/MEC Agenda Nacional identifies 470 strategic themes across 27 states to
guide funding, partnerships, and science-policy integration. Given the scale of
postgraduate programs and academic outputs, manual assessment of researcher-topic
adherence is infeasible. This project tests whether fusion of textual embeddings
from researcher production and strategic topic descriptions can be used to
automatically infer such alignment.

The core idea is to combine embeddings from two text sources — scholar production
and topic content — using various fusion operations, then train models to predict
binary affinity labels.

## Objective

The project aims to test multiple fusion operators and combinations thereof, compare
their performance across different model architectures, and identify which fusion
strategies best capture the relationship between a researcher and a strategic theme.

Key goals:

- evaluate fusion operators in isolation and in combination
- compare external fusion with fusion-inside architectures
- use Autoencoder, VAE, and Encoder models as downstream classifiers
- assess robustness through cross-validation and source-ablation experiments
- save t-SNE embeddings for visualization reuse

## Repository Structure

- `src/core/config`: shared configuration and hyperparameters
- `src/core/data`: data loading and embedding utilities
- `src/core/fusion`: fusion operators and fusion application logic
- `src/core/models`: model definitions for encoder, autoencoder, VAE, and fusion-inside variants
- `src/core/training`: training loops and trainer factories
- `src/core/experiments`: experimental phase orchestration (phase 1/2/3)
- `src/core/visualization`: plotting utilities and TSNE helpers
- `src/scripts`: entrypoint scripts for each model/architecture
- `src/analysis`: visualization runner for saved experiment outputs

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the dataset is available in `dataset/` and embeddings are available in `embeddings/`.

## Running Experiments

Each script runs the full experimental pipeline with cross-validation and saves results.
Use one of the following:

```bash
python src/scripts/main_encoder.py
python src/scripts/main_autoencoder.py
python src/scripts/main_vae.py
python src/scripts/main_autoencoder_fusion_inside.py
python src/scripts/main_vae_fusion_inside.py
python src/scripts/main_encoder_fusion_inside.py
```

The scripts save results to `results/<model_name>/`, including:

- `<model_name>_histories.pkl`
- `<model_name>_representations.npz`
- `<model_name>_tsne_embeddings.npz`
- `labels.npz`
- CSV summaries for each phase

## Visualization

After running an experiment, generate visualizations with:

```bash
python src/analysis/visualize_results.py <model_name>
```

Available model names:

- `encoder`
- `autoencoder`
- `vae`
- `encoder_fusion_inside`
- `autoencoder_fusion_inside`
- `vae_fusion_inside`

If precomputed t-SNE embeddings exist, the visualization script will load them
directly and only plot the results.

## Notes

- The project uses stratified k-fold cross-validation to evaluate fusion operators.
- Phase 1 tests isolated fusion operators.
- Phase 2 tests combinations of top operators.
- Phase 3 performs textual source ablation across different `prod`/`tema` embedding configurations.