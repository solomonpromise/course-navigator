# Course Navigator

A text-learning lab for **Modern Deep Learning & AI Engineering**, by Solomon Promise.

**[Open the deployed app](https://solomonpromise.github.io/course-navigator/)** · [Course platform](https://solomonpromise.github.io/deep-learning-lab/) · [Download the complete project](https://github.com/solomonpromise/course-navigator/archive/refs/heads/main.zip)

A learner asks a question. An adapted BERT encoder predicts one of six course topics; a separate TF-IDF index finds attributed explanations and lesson links. The project follows one task through an RNN, LSTM, GRU, our own small Transformer, frozen pretrained features, full fine-tuning and deployment. The comparison includes a strong lexical baseline.

## Start here

Download and extract the complete project. A lone notebook download does not include its data and checkpoints. Use **Python 3.12** and an isolated environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name course-navigator --display-name "Course Navigator"
jupyter lab
```

On Windows, activate with `.venv\Scripts\activate`. Select the Course Navigator kernel. Run a notebook from its first cell. Each notebook independently sets up the shared artifacts; no notebook relies on another notebook's kernel state.

In Colab, open a notebook through the app's Colab link and run its first code cell. The cell clones the project if needed. If a dependency import fails, install the requirements from the cloned project with `%pip install -r /content/course-navigator/requirements.txt`, restart the runtime, and run from the top. Package installation can take time; prepare this before class.

Default `RUN_TRAINING = False` loads actual supplied checkpoints. Set it to `True` to run training and save separate files under `artifacts/classroom/`. The reference results do not change. CPU is sufficient. Reference timing depends on hardware and is not a guarantee for a learner's machine.

## Five connected notebooks

| Notebook | Guided route | What you will do |
|---|---:|---|
| [01 · RNN](01_rnn.ipynb) | 25 min | Inspect data, prevent leakage, fit vocabulary, learn embeddings, trace gradients and train an RNN. |
| [02 · LSTM and GRU](02_lstm_gru.ipynb) | 20 min | Compute gates by hand, verify against PyTorch and compare recurrent models. |
| [03 · Our Transformer](03_transformer.ipynb) | 50 min | Implement attention, masks, residuals, positions, pooling and the classifier. |
| [04 · Pretrained adaptation](04_pretrained.ipynb) | 55 min | Load a pinned BERT checkpoint, compare frozen/full tuning, explain LoRA and inspect errors. |
| [05 · Deployment](05_deployment.ipynb) | 35 min | Define the inference contract, export and test ONNX, retrieve sources and publish the app. |

With 10 minutes for setup and 15 for questions/recap, the guided route totals **3 hours 30 minutes**. This requires prepared dependencies/checkpoints. Read every explanation and complete the extensions afterward; the timetable is not a promise to cover every cell in depth.

## Run the app locally

```sh
python -m http.server 8820 --directory app
```

Open `http://localhost:8820`. The static app uses a browser worker and a pinned WASM runtime. It loads the model on the first search, then performs inference on the device. No user text is sent to an inference service. Initial downloads and external links need a connection. No server key or inference account is needed.

## Evidence, not a predetermined winner

The frozen dataset has **1,752 actual course-prose passages from 337 source sections**: 1,201 training / 286 validation / 265 test. All windows of a section stay together. Different sections of one lesson can occur in different splits. This measures new passages within a known course, not new-course generalization. Source filenames, hashes and the exact Git revision are in `data/manifest.json`.

The separate **36 authored question probes** were frozen before training. They are not collected learner questions and are not used for fitting, checkpoint selection, temperature scaling or threshold selection. They are a small diagnostic set, not a representative product benchmark.

| Model | Passage test macro-F1 | Authored question accuracy |
|---|---:|---:|
| TF-IDF + logistic regression | 0.773 | 66.7% |
| RNN | 0.285 | 22.2% |
| LSTM | 0.477 | 30.6% |
| GRU | 0.482 | 38.9% |
| Our small Transformer | 0.523 | 41.7% |
| Frozen BERT + head | 0.491 | 38.9% |
| Fine-tuned BERT | 0.711 | 66.7% |

These are one reference CPU run, not guaranteed outcomes or statistically established rankings. Full records, curves, class reports and confusion matrices are under `artifacts/` and `figures/`.

**Why deploy BERT when TF-IDF wins passage F1?** This educational deployment demonstrates the pretrained adaptation-to-browser workflow. Its limitations and the stronger baseline are visible. A production choice would require representative question/retrieval evaluation, latency/cost comparisons and a fresh holdout. We do not claim BERT is the best production model from these results.

See [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md), [CLASSROOM_GUIDE.md](CLASSROOM_GUIDE.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `docs/verification.json`.

## Reproduce or extend the reference experiments

```sh
python scripts/train_all.py
python scripts/evaluate_export.py
python scripts/make_figures.py
python -m unittest discover -s tests -p 'test_*.py' -v
node tests/test_tokenizer.cjs
```

`train_all.py` trains models and selects checkpoints on validation only. `evaluate_export.py` records final test/probe metrics and exports the adapted model; repeated development must not optimize against that test. Keep the reference files under version control before running these commands, because these full reproduction scripts intentionally rewrite reference artifacts. Notebook experiments use a separate folder instead.

`scripts/prepare_data.py /path/to/deep-learning-platform` rebuilds the source snapshot from that repository's `origin/main`. It is an authoring operation, not required to run the bundled lab. New course notes imply a new dataset version and a new evaluation, not silent replacement.

## Publishing

The GitHub Pages workflow publishes only `app/` plus copied notebook downloads on pushes to `main`. Notebook execution never publishes. Tests and deployment records describe this release; reviewing an old record is not a replacement for verifying a changed model.

The original course notebooks and the earlier three lab folders are preserved. This project is a separate lab.
