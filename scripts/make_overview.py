"""Create the platform's short landing notebook for the five-part project."""
from pathlib import Path
import nbformat
root=Path(__file__).resolve().parents[1]
cells=[nbformat.v4.new_markdown_cell(text) for text in [
'''# Lab 4 — Course Navigator

A complete text-learning project: RNN → LSTM/GRU → our small Transformer → pretrained adaptation → a deployed lesson-finding application.

**Learning goal:** trace how text becomes learned representations, how losses update parameters, how models are evaluated and how a selected checkpoint becomes a working app.

**Prerequisites:** Modules 1–5; use this project as a guided bridge through Modules 6–8. Full setup instructions and a 3-hour-30-minute classroom route are in the project README and CLASSROOM_GUIDE.
''',
'''## 1. Open the application

Use [Course Navigator](https://solomonpromise.github.io/course-navigator/). Enter a course question; inspect the predicted topic and source excerpts. The adapted BERT model runs on your device after its initial download. Topic classification and lexical source retrieval are separate components.
''',
'''## 2. Follow five connected notebooks

1. [Learning from text with an RNN](01_rnn.ipynb): data, splits, tokens, embeddings, logits, gradients and updates.
2. [Managing memory with LSTM and GRU](02_lstm_gru.ipynb): manual gates, actual recurrent models and comparisons.
3. [Building our small Transformer](03_transformer.ipynb): explicit attention, positions, masks, pooling and training.
4. [Adapting a pretrained Transformer](04_pretrained.ipynb): frozen features, full fine-tuning, LoRA concept and evaluation.
5. [From trained model to deployed app](05_deployment.ipynb): inference contracts, ONNX checks, retrieval and publishing.

Download the complete project, not just an individual notebook, to obtain its data, code and trained artifacts. Each notebook can start from a clean kernel. Default cells load real saved checkpoints; optional training writes separate classroom artifacts.
''',
'''## 3. Understand the data and evidence

The training task uses 1,752 real course-prose passages from 337 source sections. A complete section stays in one split: 1,201 train, 286 validation and 265 test. A separate set of 36 authored questions is a diagnostic set, not collected learner traffic.

The fine-tuned BERT's passage test macro-F1 is 0.711; the TF-IDF baseline reaches 0.773. Both get 24/36 authored question topics correct. The neural model is deployed to teach pretrained adaptation and browser delivery; this does not establish its superiority over the baseline. Full metrics, source hashes and limitations are included.
''',
'''## 4. Run and review independently

Use Python 3.12, install requirements.txt in an isolated environment, select that kernel and run a notebook from the beginning. CPU is sufficient for the prepared route. In Colab, follow the setup cell and project README. To serve the app locally, run `python app.py` and open the printed localhost URL.

Read DATA_CARD.md, MODEL_CARD.md and the worked answers. Inspect errors and uncertainty as carefully as successful predictions. Real-world use requires representative consented user-question evaluation, a fresh holdout and retrieval-quality measurements.
''']]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbformat.write(nb,root/'lab.ipynb')
