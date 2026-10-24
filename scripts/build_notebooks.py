"""Author five independent teaching notebooks with visible implementation cells.

The source modules support deployment and testing. Their actual implementations
are copied into notebook code cells, so learners can read and edit the mechanisms
instead of seeing only an unexplained helper import. Figures are embedded.
"""
from pathlib import Path
import ast,base64,json,textwrap
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

def md(text):return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text):return nbf.v4.new_code_cell(textwrap.dedent(text).strip())
def source(file,*names):
    text=(ROOT/file).read_text();tree=ast.parse(text);lines=text.splitlines()
    return '\n\n'.join('\n'.join(lines[min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1:node.end_lineno]) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names)
def figure(name,alt,caption):
    cell=md(f'![{alt}](attachment:{name})\n\n*{caption}*')
    cell.attachments={name:{'image/png':base64.b64encode((ROOT/'figures'/name).read_bytes()).decode()}}
    return cell

SETUP='''
from __future__ import annotations
from pathlib import Path
import sys, json, math, random, copy, time, re
from collections import Counter
from dataclasses import dataclass

# In Colab, clone the complete project once. Locally, open this notebook from
# the extracted project folder. A lone downloaded .ipynb has no model/data files.
ROOT = Path.cwd()
if not (ROOT / "coursenav").exists():
    if "google.colab" in sys.modules:
        import subprocess
        target = Path("/content/course-navigator")
        if not target.exists():
            subprocess.run(["git", "clone", "https://github.com/solomonpromise/course-navigator.git", str(target)], check=True)
        ROOT = target
    else:
        raise FileNotFoundError("Open this notebook inside the complete Course Navigator project folder; see README.md.")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
# Optional progress widgets do not affect training; suppress only their missing-widget notice.
warnings.filterwarnings("ignore", message="IProgress not found.*")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
from IPython.display import display, Markdown
from coursenav.data import LABELS, MAX_LENGTH, words, encode, load_records, fit_vocabulary, TextDataset
from coursenav.training import (TrainConfig, seed_everything, fit, logits_for, summarize,
    dataset_for, load_reference, pretrained_tokenizer, pretrained_dataset, make_model)
seed_everything(42)
train, validation, test = (load_records(split) for split in ["train", "validation", "test"])
vocabulary = json.loads((ROOT / "artifacts/vocabulary.json").read_text())
results = json.loads((ROOT / "artifacts/results.json").read_text())["models"]
RUN_TRAINING = False  # True runs training; False loads the supplied measured checkpoint.
print("PyTorch:", torch.__version__, "| Device: CPU | Maximum tokens:", MAX_LENGTH)
print("Passages:", len(train), "train /", len(validation), "validation /", len(test), "test")
'''

def opening(number,title,minutes,goals,previous):
    return [md(f'''# Lab 4 · Course Navigator
## Notebook {number} — {title}

{previous}

**Key question:** {goals[0]}

### Learning objectives
'''+ '\n'.join('- '+goal for goal in goals[1:])+f'''

### Classroom route and independent study

Plan about **{minutes} minutes** for the guided route. Read the explanations, predict the output, then run each cell in order. Exercises and worked answers support a slower independent review. Saved results come from real CPU runs; seeds reduce variation but cannot guarantee identical numbers across machines.

### Setup

Use Python 3.12 in a dedicated environment. Download the **complete project ZIP** from the repository, extract it, install `requirements.txt`, and select that environment as your Jupyter kernel. In Colab, the first cell clones the project when needed. If imports fail, run `%pip install -r /content/course-navigator/requirements.txt` after that clone, restart the runtime and rerun from the top. Training needs no API key or paid service. The bundled base checkpoint permits offline model loading after dependencies are installed.

The default route loads supplied checkpoints. Set `RUN_TRAINING = True` to reproduce the full run into a separate classroom folder; the reference results remain unchanged. A notebook download alone does not contain its dataset and checkpoints.
'''),code(SETUP)]

def finish(filename,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (Course Navigator)',language='python',name='python3'),language_info=dict(name='python',version='3.12')))
    nbf.validate(nb);nbf.write(nb,ROOT/filename)
    print(filename,len(cells),'cells')

c=opening('1','Learning from text with an RNN',25,[
 'How do words become numbers that a neural network can learn from?',
 'Define the product, labels and dataset before choosing a model.',
 'Explain tokenization, vocabulary, padding, embeddings, logits and cross-entropy.',
 'Trace a forward pass, gradients and an optimizer update through a real text model.',
 'Separate fitting, validation, testing and retrieval; inspect the RNN’s limitations.'
],'''In Modules 1–4, we connected predictions, loss and parameter updates, then learned how to run trustworthy experiments. Module 5 introduced embeddings and recurrent networks. Here we bring those ideas together around a useful application: helping a learner find the right course explanation.''')
c += [md('''## 1. Start with the product and the target

A learner writes “Why does my validation score get worse while training improves?” The app should suggest a topic, then link to relevant explanations. Our six **labels** are broad topic categories. A label is the answer supplied during supervised training; it is not something the model discovers without guidance.

The classifier produces six scores. A separate lexical search component selects actual passages and links. This distinction matters: a topic prediction is not an answer to the question. The app does not generate new explanations.

**Example** means one passage paired with a topic. **Feature** means information used to make a prediction. Here, features are learned from token sequences. **Inference** means applying fixed learned parameters to new input.
'''),figure('01_text_pipeline.png','Text flows through tokenization, an encoder, a topic head and a separate source index.','The task head learns six course categories. Retrieved excerpts always come from the source library.'),
code('''display(pd.DataFrame({"label_id": range(len(LABELS)), "topic": LABELS}))
example = train[0]
print(example["text"])
print("Label:", LABELS[example["label"]], "| Source:", example["url"])'''),
md('''## 2. Real data, provenance and the split contract

The dataset contains **1,752 real prose passages from 337 sections** in Modules 1–8. It was extracted from the published notebooks at the Git revision recorded below. Code cells, outputs and setup sections are excluded. Text is cleaned, then divided into non-overlapping windows. A short final window is omitted when it contains fewer than 25 scratch-tokenizer words.

All windows from one section stay together. Otherwise two pieces of the same explanation could inflate apparent generalization. There are **1,201 training, 286 validation and 265 test passages**. Sections from the same lesson can still occur in different splits; this is evaluation on new sections within a known course, not new courses.

The 36 additional question probes are **authored teaching examples**, not collected learner questions. They were frozen before training and excluded from fitting, checkpoint selection and confidence-policy tuning. They help expose the gap between training on passages and receiving short questions. They are too small to establish real-world reliability.

**Data leakage** means information that should be unavailable during learning influences training or selection. Fitting a vocabulary on test text is a subtle example. Indexing all published sections for later search is a different operation: those sections are the app’s public reference library, not labels used to fit the classifier.
'''),code('''manifest = json.loads((ROOT / "data/manifest.json").read_text())
print("Source revision:", manifest["source_revision"])
groups = [{row["group"] for row in rows} for rows in [train, validation, test]]
assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
all_rows = train + validation + test
assert len({row["text_sha256"] for row in all_rows}) == len(all_rows)
display(pd.crosstab(pd.Series([LABELS[r["label"]] for r in all_rows], name="Topic"),
                    pd.Series([r["split"] for r in all_rows], name="Split")))'''),
md('''## 3. A baseline earns its place

Before building a neural network, we train a **TF-IDF + logistic regression** baseline. TF measures how often a term appears in a passage; IDF reduces the influence of terms common across the training collection. Logistic regression learns a linear decision rule on those features. It is a trained model too.

Our baseline fits its vocabulary and IDF using training rows only. Its settings were fixed before final evaluation. A strong baseline protects us from confusing complexity with progress. The saved comparison later shows that this baseline beats our fine-tuned encoder on passage macro-F1 in this reference run.
'''),code('''from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

baseline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True),
    LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42),
)
baseline.fit([row["text"] for row in train], [row["label"] for row in train])
valid_predictions = baseline.predict_proba([row["text"] for row in validation])
print("Baseline validation macro-F1:", summarize([r["label"] for r in validation], valid_predictions)["macro_f1"])'''),
md('''## 4. Token IDs are addresses, not meanings

**Tokenization** divides text into units. Our scratch tokenizer lowercases and retains word/number patterns. “Attention learns!” becomes `['attention', 'learns']`. This loses punctuation and case; every preprocessing choice carries a trade-off.

A **vocabulary** maps tokens to integer IDs. ID 47 is not inherently more meaningful than ID 3. `[PAD]` fills unused positions; `[UNK]` represents a real token absent from the training vocabulary. Only training passages establish the vocabulary. Validation and test examples are encoded with that fixed mapping.

We cap inputs at 96 scratch tokens to control compute. **Truncation** drops later tokens. **Padding** fills shorter inputs; a true-length tensor tells recurrence where each sequence ends. The deployed pretrained model has its own tokenizer and reserves two of its 96 positions for special tokens.
'''),code(source('coursenav/data.py','words','fit_vocabulary','encode')),
code('''assert fit_vocabulary(train) == vocabulary
text = "Attention learns useful relationships between tokens."
ids, length = encode(text, vocabulary)
print("Words:", words(text))
print("IDs:", ids[:length], "| Real length:", length)
print("Padding positions:", len(ids) - length)
unknown_fraction = np.mean([word not in vocabulary for row in validation for word in words(row["text"])])
print("Validation unknown-word fraction:", round(float(unknown_fraction), 3))'''),
md('''## 5. Embeddings are learned rows

An **embedding table** is a parameter matrix with one row per vocabulary entry. Looking up a token ID selects its row. The row begins as random numbers for our scratch models. Backpropagation adjusts rows used by training examples so that their downstream predictions improve.

If the vocabulary has V entries and each vector has D components, the matrix has V × D parameters. A batch of IDs shaped `(B, T)` becomes vectors shaped `(B, T, D)`. B is the number of examples; T is the number of time steps; D is the embedding width.

Unlike a pretrained language encoder, these embeddings see only this small course corpus. Similarity in a two-dimensional projection is an illustration, not proof of general language understanding.
'''),code('''embedding = nn.Embedding(len(vocabulary), 64, padding_idx=0)
sample_ids = torch.tensor([ids], dtype=torch.long)
print("Table:", tuple(embedding.weight.shape))
print("Batch of vectors:", tuple(embedding(sample_ids).shape))
print("Trainable table entries:", embedding.weight.numel())'''),figure('07_embeddings.png','A PCA projection of actual learned GRU embedding rows.','These are measured embedding vectors from Notebook 2’s saved GRU. Projection discards information.'),
md(r'''## 6. Read the sequence with a recurrent state

An RNN repeatedly applies the same update:

$$h_t=\tanh(W_xx_t+W_hh_{t-1}+b).$$

The **hidden state** is a vector carrying information forward. Parameter sharing means the same matrices operate at every time step. Earlier words can influence later states through the chain of updates. A final linear **head** converts the last valid state into six topic logits.

We use packed sequences so padding does not become extra fake words. PyTorch handles the recurrent loop efficiently; the mathematical operation remains the update above. This is a unidirectional, one-layer baseline. Its weaknesses do not establish that every recurrent architecture is unsuitable for text.
'''),code(source('coursenav/models.py','RecurrentClassifier')),
code('''seed_everything()
rnn = RecurrentClassifier(len(vocabulary), kind="RNN")
batch = next(iter(DataLoader(TextDataset(train, vocabulary), batch_size=8)))
batch_ids, lengths, targets = batch
logits = rnn(batch_ids, lengths)
print("IDs:", tuple(batch_ids.shape), "Lengths:", tuple(lengths.shape))
print("Logits:", tuple(logits.shape), "Targets:", targets.tolist())
assert logits.shape == (8, len(LABELS))'''),
md(r'''## 7. Loss gives learning a direction

A **logit** is an unrestricted score. Softmax converts six logits into positive numbers summing to one. For a correct label y, cross-entropy is $-\log p_y$. Assigning little probability to the true label produces a large loss.

`CrossEntropyLoss` combines log-softmax and negative log likelihood in a numerically stable operation. Pass raw logits to it. Applying softmax first changes the calculation and loses numerical stability.

`backward()` computes derivatives, not updated weights. A **gradient** tells us how the loss would change for a small parameter change. The optimizer performs the update. AdamW maintains moving gradient statistics and applies decoupled weight decay; its learning rate controls the scale of those updates.
'''),code('''toy_logits = torch.tensor([[1.2, -0.3, 0.4, 0.1, -0.5, 0.2]])
toy_target = torch.tensor([0])
probabilities = toy_logits.softmax(-1)
manual_loss = -probabilities[0, 0].log()
library_loss = nn.functional.cross_entropy(toy_logits, toy_target)
print("Probabilities:", probabilities.numpy().round(3))
print("Manual / stable library loss:", manual_loss.item(), library_loss.item())
assert torch.allclose(manual_loss, library_loss)'''),
md('''## 8. Watch one actual parameter update

This demonstration uses an independent model copy, so it cannot corrupt the supplied reference checkpoint. We clear old gradients, compute logits and loss, backpropagate, inspect a gradient, clip its norm and update parameters. Clipping limits exploding updates; it does not solve vanishing gradients.

Predict first: after `backward()`, have the weights already changed? The answer is no. After `step()`, they should change if their gradient is nonzero.
'''),code('''demo = copy.deepcopy(rnn)
optimizer = torch.optim.AdamW(demo.parameters(), lr=0.002, weight_decay=0.01)
before = demo.head.weight.detach().clone()
optimizer.zero_grad(set_to_none=True)
loss = nn.functional.cross_entropy(demo(batch_ids, lengths), targets)
loss.backward()
print("Head gradient norm:", float(demo.head.weight.grad.norm()))
assert torch.equal(before, demo.head.weight.detach())
torch.nn.utils.clip_grad_norm_(demo.parameters(), max_norm=1.0)
optimizer.step()
print("Largest head-weight change:", float((demo.head.weight.detach()-before).abs().max()))'''),
md('''## 9. The complete training loop, with validation checkpoints

An **epoch** is one pass through the training rows. A **minibatch** supplies one noisy loss estimate. We rebalance cross-entropy using training class counts, shuffle training batches with a seeded generator, and retain the epoch with the highest validation macro-F1.

**Precision** asks how often predictions for a class are correct. **Recall** asks how many true examples of that class were found. **F1** combines precision and recall. **Macro-F1** computes F1 separately for each topic, then averages, giving each topic equal weight.

`model.train()` enables dropout; `model.eval()` disables it. `inference_mode()` disables gradient tracking during evaluation. These solve different problems. Early stopping uses validation, never test performance. Read this loop slowly: it is the learning mechanism from Module 2 applied to language.
'''),code(source('coursenav/training.py','TrainConfig','logits_for','summarize','fit')),
code('''if RUN_TRAINING:
    seed_everything()
    rnn, history, seconds = fit(RecurrentClassifier(len(vocabulary), "RNN"),
        TextDataset(train, vocabulary), TextDataset(validation, vocabulary), TrainConfig())
    output = ROOT / "artifacts/classroom"
    output.mkdir(exist_ok=True)
    torch.save(rnn.state_dict(), output / "rnn.pt")
else:
    rnn.load_state_dict(torch.load(ROOT / "artifacts/rnn.pt", map_location="cpu", weights_only=True))
    history = next(row for row in results if row["name"] == "rnn")["history"]
rnn.eval()
display(pd.DataFrame(history).tail())'''),
figure('02_learning_curves.png','Actual training losses and validation macro-F1 for the six neural experiments.','The reference curves are measurements, not expected values to force a new run to match.'),
md('''## 10. Read the result without overclaiming

The saved RNN’s passage test macro-F1 is about **0.285**, and it answers 8 of 36 authored topic probes correctly. That is weak for a six-topic routing task. Its training loss falls much further than its validation performance improves, warning us about limited generalization.

Possible contributors include the small dataset, a final-state bottleneck, long sequences, initialization and this training configuration. These results alone do not isolate which cause dominates. We did not tune on the test set to make the RNN look better. Notebook 2 investigates architectures designed to manage state more carefully.
'''),code('''row = next(row for row in results if row["name"] == "rnn")
display(pd.DataFrame(row["test"]["classification_report"]).T)
probe = "Why does a model need an activation function?"
encoded, length = encode(probe, vocabulary)
with torch.inference_mode():
    probability = rnn(torch.tensor([encoded]), torch.tensor([length])).softmax(-1)[0]
display(pd.Series(probability.numpy(), index=LABELS).sort_values(ascending=False))'''),
md('''## 11. Try it, then check your reasoning

1. Predict the effect of replacing every word with `[UNK]`. Test on a copy of an encoded batch. Why can the network still emit a confident score?
2. Change 96 to a smaller maximum length in an isolated experiment. What training cost and information-loss trade-offs should you measure?
3. Why does rebuilding the vocabulary after loading a checkpoint break the model even when its shape stays the same?
4. Why is a question-probe score different from performance on actual learners?

<details><summary>Worked answers</summary>

1. Topic-specific word evidence disappears; position, length and biases can still drive logits. Confidence is not a proof of understanding.
2. Shorter inputs reduce recurrent steps and memory, but may remove decisive context. Refit on training, select on validation, and reserve a new test for repeated development.
3. Rows now refer to different words. Weight shapes are only part of the inference contract; the exact ID mapping matters.
4. Authored questions are small, curated and may differ from real spelling, ambiguity, language and topic frequencies. A representative evaluation needs consented real inputs and a separate protocol.

</details>

**Next:** [Notebook 2 — LSTM and GRU](02_lstm_gru.ipynb). Keep the same data and labels; change how information moves through time.
''')]
finish('01_rnn.ipynb',c)

c=opening('2','Managing memory with LSTM and GRU',20,[
 'How can a recurrent model learn what to preserve, replace or expose?',
 'Interpret sigmoid gates and separate hidden state from cell state.',
 'Compute one LSTM and GRU update and verify each against PyTorch.',
 'Train or load gated models under the shared evaluation contract.',
 'Compare accuracy, parameter counts and failure cases without declaring a universal winner.'
],'''Notebook 1 showed the whole learning pipeline and a weak simple-RNN reference result. Module 5.4 introduced the reason to consider gates. Here we keep tokenization, labels and splits fixed and open up the state-update mechanism.''')
c += [md(r'''## 1. Why repeated state updates can become difficult

Backpropagation through time multiplies derivatives across recurrent steps. Products can shrink or grow, making distant dependencies difficult to learn. A gradient problem is about the learning signal, not a claim that the input words disappeared from memory in a literal human sense.

A **gate** is a learned vector of values between zero and one, usually produced with sigmoid. Multiplying by a gate softly controls a pathway. A gate is computed from the current input and state; it is not a rule we manually label for every word.

Sigmoid(large positive) approaches one; sigmoid(large negative) approaches zero. Values in the middle pass partial amounts. All gates are trained through the final prediction loss.
'''),code('''values = torch.tensor([-5., -1., 0., 1., 5.])
display(pd.DataFrame({"input": values.numpy(), "sigmoid gate": values.sigmoid().numpy()}))'''),figure('05_memory_mechanisms.png','RNN, LSTM and GRU state-update equations.','The GRU equation uses PyTorch’s convention: a large update gate preserves more of the old hidden state.'),
md(r'''## 2. LSTM: separate storage from exposure

An **LSTM**, or long short-term memory network, carries a cell state $c_t$ and a hidden state $h_t$:

$$f_t=\sigma(a_f),\quad i_t=\sigma(a_i),\quad g_t=\tanh(a_g),\quad o_t=\sigma(a_o)$$
$$c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t).$$

Each preactivation a is a learned linear combination of input and previous hidden state. The **forget gate** scales old cell contents. The **input gate** scales candidate contents. The **output gate** controls what the hidden state exposes. The symbol ⊙ means multiply matching vector components.

The additive cell-state path can support learning longer dependencies. It does not guarantee perfect memory or eliminate every optimization problem. PyTorch packs its four gate projections in the order input, forget, candidate, output. We verify that convention numerically instead of relying on a diagram.
'''),code('''seed_everything()
cell = nn.LSTMCell(input_size=3, hidden_size=4)
x, h_old, c_old = torch.randn(1, 3), torch.randn(1, 4), torch.randn(1, 4)
gates = x @ cell.weight_ih.T + cell.bias_ih + h_old @ cell.weight_hh.T + cell.bias_hh
i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)
i, f, g, o = i_raw.sigmoid(), f_raw.sigmoid(), g_raw.tanh(), o_raw.sigmoid()
c_new = f * c_old + i * g
h_new = o * c_new.tanh()
h_library, c_library = cell(x, (h_old, c_old))
assert torch.allclose(h_new, h_library, atol=1e-6)
assert torch.allclose(c_new, c_library, atol=1e-6)
display(pd.DataFrame({"forget": f.detach().numpy()[0], "input": i.detach().numpy()[0],
                     "output": o.detach().numpy()[0], "new cell": c_new.detach().numpy()[0]}))'''),
md(r'''## 3. GRU: two gates and one main state

A **gated recurrent unit** uses reset r and update z gates. In PyTorch’s convention:

$$r=\sigma(W_{ir}x+b_{ir}+W_{hr}h+b_{hr}),\quad z=\sigma(W_{iz}x+b_{iz}+W_{hz}h+b_{hz})$$
$$n=\tanh(W_{in}x+b_{in}+r\odot(W_{hn}h+b_{hn}))$$
$$h_{new}=(1-z)\odot n+z\odot h.$$

The reset gate controls how old state contributes to the candidate. A large update gate preserves more old state in this formulation. Some texts define z with the opposite convention, so always read the equation. PyTorch applies its reset gate after the hidden-state projection for the candidate; moving it before the matrix multiplication changes the model.
'''),code('''cell = nn.GRUCell(input_size=3, hidden_size=4)
x, old = torch.randn(1, 3), torch.randn(1, 4)
input_parts = (x @ cell.weight_ih.T + cell.bias_ih).chunk(3, -1)
hidden_parts = (old @ cell.weight_hh.T + cell.bias_hh).chunk(3, -1)
reset = (input_parts[0] + hidden_parts[0]).sigmoid()
update = (input_parts[1] + hidden_parts[1]).sigmoid()
candidate = (input_parts[2] + reset * hidden_parts[2]).tanh()
manual = (1 - update) * candidate + update * old
assert torch.allclose(manual, cell(x, old), atol=1e-6)
print("GRU manual/library maximum difference:", float((manual-cell(x, old)).abs().max().detach()))'''),
md('''## 4. Put gates into the complete text classifier

The embedding layer and topic head remain unchanged. We replace the recurrent mechanism and still extract the last valid hidden state. An LSTM returns both hidden and cell states; the head reads the hidden state. Packed input prevents trailing padding from altering that final state.

The same hidden width does not mean equal parameter count: gate projections add parameters. This comparison changes architecture and capacity together. All scratch models still start from random embeddings and train only on the course passages.
'''),code(source('coursenav/models.py','RecurrentClassifier')),
code('''models = {}
for kind in ["RNN", "LSTM", "GRU"]:
    model = RecurrentClassifier(len(vocabulary), kind)
    model.load_state_dict(torch.load(ROOT / f"artifacts/{kind.lower()}.pt", map_location="cpu", weights_only=True))
    models[kind] = model.eval()
display(pd.DataFrame([{"model": kind, "parameters": sum(p.numel() for p in model.parameters())}
                      for kind, model in models.items()]))'''),
md('''## 5. Reproduce a gated run or inspect the saved evidence

The training loop is explained fully in Notebook 1 and available in `coursenav/training.py`. We reuse it so the split, metrics and checkpoint policy remain consistent. Set the flag to train a GRU; its output goes into `artifacts/classroom`, leaving the reference run intact.
'''),code('''if RUN_TRAINING:
    seed_everything()
    experiment, history, seconds = fit(RecurrentClassifier(len(vocabulary), "GRU"),
        TextDataset(train, vocabulary), TextDataset(validation, vocabulary), TrainConfig())
    output = ROOT / "artifacts/classroom"; output.mkdir(exist_ok=True)
    torch.save(experiment.state_dict(), output / "gru.pt")
comparison = [{"model": row["name"], "parameters": row["parameters"],
    "best validation F1": max(epoch["validation_macro_f1"] for epoch in row["history"]),
    "test macro-F1": row["test"]["macro_f1"], "question accuracy": row["questions"]["accuracy"]}
    for row in results if row["name"] in ["rnn", "lstm", "gru"]]
display(pd.DataFrame(comparison).set_index("model"))'''),
md('''## 6. Look at disagreements, not just averages

The saved LSTM and GRU passage test macro-F1 scores are about **0.477 and 0.482**. A difference that small in one run is not strong evidence that the GRU is inherently better. Seed variation, uncertainty intervals and repeated data protocols would be needed for a stronger claim.

The question below is an exploratory input, not a new accuracy measurement. A model can predict a plausible topic with high confidence while still missing the learner’s intended meaning. Examine every score and follow the source material.
'''),code('''question = "How can a network keep information from earlier words?"
ids, length = encode(question, vocabulary)
rows = []
with torch.inference_mode():
    for name, model in models.items():
        p = model(torch.tensor([ids]), torch.tensor([length])).softmax(-1)[0]
        rows.append({"model": name, "prediction": LABELS[int(p.argmax())], "raw confidence": float(p.max())})
display(pd.DataFrame(rows))'''),
md('''## 7. What gates improve, and what remains

Gates create learnable routes for preserving information; they do not make the model read all tokens simultaneously. These recurrent models still process time steps sequentially. Our final-state head also compresses the entire input into one vector.

Attention offers another mechanism: each token can gather information directly from other allowed tokens. That changes the information path, while introducing quadratic attention-score cost in sequence length. Notebook 3 builds that mechanism explicitly.

**Try it.** Force an LSTM forget gate near zero or one in the manual example. Predict what happens to the old cell’s contribution. Then test how a GRU update gate near one affects its old state.

<details><summary>Worked answer</summary>

Forget near zero suppresses the previous cell term; near one preserves it before adding new candidate content. For the PyTorch GRU convention used here, update near one retains the previous hidden state. Gates control pathways component by component, not whole sentences with a single Boolean switch.

</details>

**Next:** [Notebook 3 — Our small Transformer](03_transformer.ipynb).
''')]
finish('02_lstm_gru.ipynb',c)

c=opening('3','Building our small Transformer',50,[
 'How can every token gather useful context without a recurrent time-step loop?',
 'Trace queries, keys, values, scaling, softmax and multiple heads.',
 'Distinguish padding masks from causal masks and classification from generation.',
 'Implement an encoder block, positional embeddings and masked pooling.',
 'Validate attention against PyTorch, train the model and inspect real attention maps.'
],'''Module 6 develops the Transformer family and language-model training. Here we build an encoder for the same classification task used in Notebooks 1–2. The attention calculations carry across; the objective and masking choices follow the product we are building.''')
c += [md(r'''## 1. Match the architecture to the objective

Our task is **sequence classification**: read a complete question or passage, then predict a course topic. An encoder may use both earlier and later tokens. A next-token language model has a different objective and must hide future tokens during prediction.

Our small Transformer is randomly initialized and learns only from the course training split. It has two blocks, width 64 and four heads. It is deliberately small enough to inspect on a CPU. It does not have the language experience of a pretrained encoder.

Recall Notebook 2’s bottleneck: recurrent updates carry information through time. Self-attention instead constructs direct token-to-token connections. That shortens the path between distant tokens but does not guarantee better performance on a small dataset.
'''),
md(r'''## 2. Queries, keys and values are learned projections

For each input vector x, three linear projections create a **query** q, **key** k and **value** v. Queries and keys determine relevance; values carry the information being combined.

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad O=AV.$$

Each row of A distributes one query’s weight across allowed keys. The square-root scaling keeps dot-product magnitudes from growing simply because the head dimension grows. The mask M assigns a large negative score to disallowed keys before softmax.

These are learned computations, not human-supplied explanations. A high attention weight is not a causal proof that a word determined the prediction.
'''),code('''# Small invented numbers let us check the arithmetic before using learned weights.
q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
k = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])
v = torch.tensor([[2., 0.], [0., 3.], [1., 1.]])
scores = q @ k.T / math.sqrt(q.shape[-1])
weights = scores.softmax(-1)
output = weights @ v
display(pd.DataFrame(scores.numpy()).round(3))
display(pd.DataFrame(weights.numpy()).round(3))
print("Each attention row sums to:", weights.sum(-1))
print("Mixed values:", output)'''),
md('''## 3. Two masks, two different reasons

A **padding mask** prevents real tokens from reading artificial PAD keys. A **causal mask** prevents position t from reading future positions. For this encoder, every real token may read every other real token; only padding keys are hidden.

Why do we mask keys rather than merely zeroing PAD embeddings? A zero vector can still receive attention probability, and learned projections can add biases. Why exclude padded query outputs from pooling? A padded query can still read real keys and become nonzero. Both issues must be handled.

The causal example below connects to Module 6’s generative models. Applying this triangular mask alone would not turn our classifier into a trained language model: we would also need a vocabulary-sized output head, shifted token targets and next-token loss.
'''),code('''causal = torch.ones(3, 3, dtype=torch.bool).tril()
causal_weights = scores.masked_fill(~causal, -1e4).softmax(-1)
print("Encoder attention:", weights.round(decimals=3))
print("Causal attention:", causal_weights.round(decimals=3))
assert torch.equal(causal_weights.triu(1), torch.zeros_like(causal_weights))'''),
md('''## 4. Implement multi-head attention

With width D=64 and H=4 heads, each head receives 16 components. We project Q/K/V, reshape `(B,T,D)` into `(B,H,T,D/H)`, compute per-head attention, then concatenate the head outputs and project again.

Each head has learned projection parameters; it is not assigned a fixed role such as “grammar” or “meaning.” Inspecting a head may suggest patterns, but these labels need evidence.

Attention creates a `(B,H,T,T)` score tensor. Doubling T approximately quadruples that tensor’s size. This is one reason to limit sequence length and batch size. Training can compute the token interactions in parallel; autoregressive generation still has sequential token-generation dependencies.
'''),code(source('coursenav/models.py','SelfAttention')),
code('''seed_everything()
attention = SelfAttention(width=64, heads=4).eval()
x = torch.randn(2, 7, 64)
mask = torch.tensor([[1,1,1,1,0,0,0], [1,1,1,1,1,1,1]], dtype=torch.bool)
actual, maps = attention(x, mask, inspect=True)
q, k, v = [part.reshape(2,7,4,16).transpose(1,2) for part in attention.qkv(x).chunk(3,-1)]
reference = nn.functional.scaled_dot_product_attention(q, k, v,
    attn_mask=mask[:,None,None,:], dropout_p=0.0)
reference = attention.output(reference.transpose(1,2).reshape(2,7,64))
assert torch.allclose(actual, reference, atol=1e-6)
assert float(maps[0,:,:,4:].abs().max().detach()) == 0.0
print("Maximum implementation/reference difference:", float((actual-reference).abs().max().detach()))'''),
md('''## 5. Build a block around attention

The **residual connection** adds a layer’s update to its input, preserving a direct information and gradient path. **Layer normalization** rescales a token’s feature components using learned scale and bias. Our block normalizes before each sublayer, a pre-normalization design.

The **feed-forward network** expands each token from D to 4D, applies GELU, then projects back. Its weights are shared across token positions; it does not itself mix tokens. Attention does the token mixing.

**Dropout** randomly suppresses selected activations during training. It is disabled at inference. Normalization, residuals and dropout have distinct jobs; none can repair poor data or a leaking evaluation.
'''),code(source('coursenav/models.py','EncoderBlock')),
md('''## 6. Positions and a sequence-level decision

Attention by itself does not encode order. We add a learned positional vector to each word embedding. A position table has an explicit maximum length; truncation and the exported inference contract must respect it.

After two blocks, **masked mean pooling** averages real token states into one vector. The head maps that vector to topic logits. We exclude PAD outputs even though PAD keys were already masked. A `[CLS]`-based head is another design; our mean-pooling choice is fixed before evaluation.
'''),code(source('coursenav/models.py','TinyTransformer')),
code('''seed_everything()
transformer = TinyTransformer(len(vocabulary))
transformer.load_state_dict(torch.load(ROOT / "artifacts/transformer.pt", map_location="cpu", weights_only=True))
transformer.eval()
print("Parameters:", sum(p.numel() for p in transformer.parameters()))
batch = next(iter(DataLoader(TextDataset(validation, vocabulary), batch_size=4)))
with torch.inference_mode():
    prediction = transformer(batch[0], batch[1])
print("Output shape:", tuple(prediction.shape))
assert prediction.shape == (4, len(LABELS))'''),
md('''## 7. Train with the same learning contract

Token IDs select embeddings; positions are added; attention and feed-forward blocks build context; pooling produces a sequence representation; the head emits logits. Cross-entropy compares logits with known topics. Backpropagation updates all trainable components, including embeddings and positions.

The reference loop and optimizer are the same as Notebook 1. The models have different parameter counts and optimization behavior, so equal epochs do not imply equal computation or a perfectly controlled study.
'''),code('''if RUN_TRAINING:
    seed_everything()
    transformer, history, seconds = fit(TinyTransformer(len(vocabulary)),
        TextDataset(train, vocabulary), TextDataset(validation, vocabulary), TrainConfig())
    output = ROOT / "artifacts/classroom"; output.mkdir(exist_ok=True)
    torch.save(transformer.state_dict(), output / "transformer.pt")
record = next(row for row in results if row["name"] == "transformer")
display(pd.DataFrame(record["history"]))'''),
md('''## 8. Inspect an actual attention map

The figure below uses learned weights from the saved scratch Transformer. A row answers “where did this query gather values in this head?” Different heads and layers can differ. The map is an internal measurement, not a full explanation of the final prediction.

Try a sentence about a different topic, trace it through the same layer and compare. Keep the model in evaluation mode so dropout does not randomly alter the map.
'''),figure('06_attention.png','Learned attention weights for a sentence about attention.','First block, first head. Padding columns are excluded from this display.'),code('''text = "attention lets each token read information from other tokens"
ids, length = encode(text, vocabulary)
input_ids = torch.tensor([ids]); mask = input_ids.ne(0)
with torch.inference_mode():
    x = transformer.embedding(input_ids) + transformer.position(torch.arange(MAX_LENGTH))[None,:,:]
    _, attention_maps = transformer.blocks[0].attention(transformer.blocks[0].norm1(x), mask, inspect=True)
print("Attention tensor:", tuple(attention_maps.shape))
print("Real-key row sums:", attention_maps[0,0,:length,:length].sum(-1))'''),
md('''## 9. What this run establishes

Our scratch encoder reaches about **0.523 passage test macro-F1**, compared with about 0.669 at its best validation epoch. The validation/test gap matters. It is evidence that a single validation score can overstate the generalization we later observe.

The result exceeds this simple RNN’s score, but it does not prove that attention universally beats recurrence. The encoder also differs in pooling, parameterization and depth. The TF-IDF baseline remains stronger on this dataset. More data, different architectures and repeated seeds would be new experiments, requiring careful validation and fresh held-out assessment if we repeatedly develop against the current test.

**Check yourself.**

1. Why does padding masking not remove the need for masked pooling?
2. What changes if we apply a causal mask to a complete question?
3. What else must change to train next-token generation?
4. Why does more attention capacity not guarantee better test performance?

<details><summary>Worked answers</summary>

1. Padded queries can gather real values and become nonzero; pooling must still exclude their outputs.
2. Earlier positions lose access to later context. This changes the information available to the encoder and needs a deliberate objective/architecture decision.
3. Predict token-vocabulary logits at every position, shift targets, mask future tokens and padding appropriately, and train a language-model objective. Generation then appends sampled or selected tokens step by step.
4. Capacity can fit training details the test does not share. Data quality, supervision, optimization, regularization and evaluation design remain decisive.

</details>

**Next:** [Notebook 4 — Adapting a pretrained Transformer](04_pretrained.ipynb). We reuse representations learned from a much larger language corpus.
''')]
finish('03_transformer.ipynb',c)

c=opening('4','Adapting a pretrained Transformer',55,[
 'What changes when useful language representations already exist?',
 'Read a model card and pin an exact pretrained checkpoint and tokenizer.',
 'Compare frozen features with full fine-tuning using explicit PyTorch loops.',
 'Explain learning-rate choice, catastrophic forgetting and low-rank adaptation.',
 'Evaluate the reference models honestly and select a confidence policy on validation only.'
],'''Notebook 3 built and trained an encoder from random initialization. Modules 7–8 explain how pretraining changes the starting point and how to adapt an existing model. Here the architecture is still a Transformer, but its parameters already contain information learned before our course task.''')
c += [md('''## 1. Choose a model for the constraints

We use Google’s **BERT-Tiny**, `google/bert_uncased_L-2_H-128_A-2`: two layers, hidden width 128, two attention heads and an uncased WordPiece vocabulary. The model card identifies an Apache-2.0 license and describes its compact pretrained design. A small encoder keeps this class practical on a CPU and permits an approximately 18 MB browser model.

A **checkpoint** is a saved parameter state plus configuration. A model identifier can change over time, so we also pin its revision. A **model card** describes intended uses, training background, evaluation and limitations. Permission to use a checkpoint does not establish that it is suitable for our task.

BERT’s original pretraining objectives differ from our six-topic classification objective. Pretraining does not magically teach it these labels. We attach and train a new topic head. Our comparison also changes parameter count, tokenizer and prior data, so it is not a pure architecture experiment.

Sources: [Google model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2), [BERT documentation](https://huggingface.co/docs/transformers/model_doc/bert).
'''),code('''from coursenav.training import MODEL_ID, MODEL_REVISION
print("Model:", MODEL_ID)
print("Pinned revision:", MODEL_REVISION)
print(json.loads((ROOT / "artifacts/pretrained-source/config.json").read_text()))'''),
md('''## 2. Keep the tokenizer and model together

**WordPiece** can split an unfamiliar word into known subword pieces. For example, a rare technical term may become a root and several `##` continuations. Its vocabulary is fixed by the pretrained checkpoint. Replacing it with our scratch vocabulary would send the wrong embedding rows into the model.

`[CLS]` and `[SEP]` mark sequence boundaries. `[PAD]` fills unused positions. `attention_mask=1` marks real/special tokens, and zero marks padding. `token_type_ids` distinguish sentence segments in BERT; our single-input examples use zero throughout.

Our 96-position contract leaves at most **94 content WordPieces**, plus two boundary tokens. This is not the same as 94 words. Tokenization can truncate important late context, so the app displays a truncation notice.

The constructor below uses the explicit vocabulary API tested in our pinned Transformers version. Its assertions catch a silent five-special-token fallback: an API mismatch must fail before we accept any training result.
'''),code(source('coursenav/training.py','pretrained_tokenizer','pretrained_dataset')),
code('''tokenizer = pretrained_tokenizer()
sample = "Fine-tuning a pretrained encoder changes its representations."
encoded = tokenizer(sample, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
print("Pieces:", tokenizer.tokenize(sample))
print("IDs:", encoded["input_ids"][0,:16].tolist())
print("Real/special tokens:", int(encoded["attention_mask"].sum()))
assert tokenizer.vocab_size == 30522 and tokenizer.cls_token_id == 101'''),
md('''## 3. Add a task head to contextual representations

The encoder returns `(B,T,128)` contextual states. Unlike the static embedding lookup from Notebook 1, a token’s final state depends on allowed surrounding tokens. Masked mean pooling turns the sequence into one 128-component vector. A new linear head maps it to six logits.

The original masked-language-model head is not our classifier and is not reused. Loading the encoder may report unused pretraining-head keys; that is expected for this deliberate architecture change. A checkpoint mismatch within our own six-topic model would be a real error.
'''),code(source('coursenav/models.py','PretrainedClassifier')),
md('''## 4. Frozen features: learn the head first

Freezing sets encoder parameters’ `requires_grad=False`. Backpropagation still trains the head. We also keep the encoder in evaluation mode during head training so dropout does not randomly change its frozen features.

This is a useful baseline: how far can existing representations take us with minimal adaptation? Its best reference validation macro-F1 is about 0.508. Frozen inference is not free—the encoder still performs its forward computations—but fewer parameters require gradients and optimizer state.
'''),code('''frozen = make_model("frozen")
display(pd.DataFrame([
    {"component": "encoder", "trainable": sum(p.numel() for p in frozen.encoder.parameters() if p.requires_grad)},
    {"component": "head", "trainable": sum(p.numel() for p in frozen.head.parameters() if p.requires_grad)},
]))
if RUN_TRAINING:
    seed_everything()
    frozen, history, seconds = fit(frozen, pretrained_dataset(train), pretrained_dataset(validation),
        TrainConfig(epochs=16, learning_rate=.008), frozen=True)
    output = ROOT / "artifacts/classroom"; output.mkdir(exist_ok=True)
    torch.save(frozen.state_dict(), output / "frozen.pt")'''),
md('''## 5. Full fine-tuning: update the encoder too

**Fine-tuning** starts from pretrained weights and updates them on task data. This lets contextual representations adapt to our vocabulary and distinctions. It can also damage useful prior behavior—**catastrophic forgetting**—or overfit a small dataset.

Our predeclared reference configuration uses AdamW, learning rate 0.0002, weight decay 0.01, batch size 32, up to 12 epochs, gradient clipping and validation checkpoint selection. This rate is an experimental choice for this compact model, not a universal BERT recommendation. A larger model or different task needs its own validation protocol.

No test score selected the checkpoint or these settings. We never claim that the model acquired general teaching ability: it learned a narrow topic-routing task.
'''),code('''seed_everything()
adapted = make_model("finetuned")
print("Trainable parameters:", sum(p.numel() for p in adapted.parameters() if p.requires_grad))
if RUN_TRAINING:
    adapted, history, seconds = fit(adapted, pretrained_dataset(train), pretrained_dataset(validation),
        TrainConfig(epochs=12, learning_rate=.0002))
    output = ROOT / "artifacts/classroom"; output.mkdir(exist_ok=True)
    torch.save(adapted.state_dict(), output / "finetuned.pt")
else:
    adapted.load_state_dict(torch.load(ROOT / "artifacts/finetuned.pt", map_location="cpu", weights_only=True))
adapted.eval()
record = next(row for row in results if row["name"] == "finetuned")
display(pd.DataFrame(record["history"]))'''),
md(r'''## 6. Where LoRA fits

**Parameter-efficient fine-tuning** updates a restricted set of parameters. LoRA freezes a weight W and learns a low-rank update:

$$W_{adapted}=W+\frac{\alpha}{r}BA.$$

For a matrix with output width m and input width n, a full update has mn parameters. Rank-r factors have r(m+n). A small rank constrains the update’s capacity. This can save trainable memory and optimizer state, though it does not remove the base model’s inference cost.

We demonstrate the mechanism below. **This is not a trained LoRA experiment**, and it contributes no score to the results table. Full fine-tuning is feasible for our small checkpoint; running an additional LoRA study belongs to further learning. QLoRA additionally uses a quantized base model and introduces numerical/hardware trade-offs beyond this CPU classroom route.
'''),code('''# A shape-and-initialization demonstration, not a reported training run.
in_features, out_features, rank, alpha = 128, 128, 4, 8
base = nn.Linear(in_features, out_features, bias=False)
for parameter in base.parameters():
    parameter.requires_grad = False
A = nn.Parameter(torch.randn(rank, in_features) * .01)
B = nn.Parameter(torch.zeros(out_features, rank))
x = torch.randn(2, in_features)
original = base(x)
with_adapter = original + (alpha/rank) * (x @ A.T @ B.T)
assert torch.equal(original, with_adapter)  # zero B makes the initial update zero
print("Full matrix parameters:", in_features*out_features)
print("Adapter parameters:", A.numel()+B.numel())'''),
md('''## 7. Read every model’s final evidence

The fine-tuned model reaches approximately **0.711 passage test macro-F1** and **24/36 authored questions correct**. The TF-IDF baseline reaches approximately **0.773** and **24/36**, respectively. The fine-tuned encoder wins among our neural experiments; the simpler baseline wins the passage comparison.

**Deployment decision for this lab:** we deploy fine-tuned BERT to demonstrate the complete pretrained-model adaptation and browser export workflow requested for class. We disclose its limits and keep the baseline visible. A production routing decision based only on this evidence would favor the baseline’s measured passage result, then evaluate latency, confidence, real question performance and maintenance. We do not present this educational choice as empirical superiority.

One seed and a small authored question set do not support strong statistical rankings. Classes overlap: a question about fine-tuning can also mention pretrained applications. Some labels are legitimately ambiguous; inspect mistakes rather than hiding them in a single average.
'''),figure('03_model_comparison.png','All seven models compared on held-out passages and authored questions.','Different metrics answer different questions. Neither bar measures actual learner satisfaction.'),figure('04_confusion_matrix.png','The fine-tuned model’s held-out passage confusion matrix.','Rows are true topics; columns are predicted topics. Off-diagonal cells are the mistakes we must investigate.'),
code('''summary = [{"model": row["name"], "passage macro-F1": row["test"]["macro_f1"],
            "question accuracy": row["questions"]["accuracy"], "training seconds": row["train_seconds"]} for row in results]
display(pd.DataFrame(summary).set_index("model"))'''),
md('''## 8. Confidence is a separate engineering problem

Softmax scores often become overconfident after fitting. **Temperature scaling** divides logits by a positive scalar before softmax. It changes score sharpness without changing the winning class. The scalar is fitted on validation labels only.

The saved temperature is about 2.364. A validation-selected threshold near 0.625 marks tentative predictions; in validation it retains about 74.5% of examples with roughly 81.2% accuracy among retained examples. These are validation measurements used to choose the policy, so they are not unbiased deployment guarantees.

The app also checks whether lexical retrieval finds meaningful overlap. This is a heuristic for weak matches, not a validated out-of-domain detector. Confident unrelated questions can still fool the model. We expose uncertainty and source excerpts rather than inventing an answer.
'''),code('''policy = json.loads((ROOT / "artifacts/policy.json").read_text())
sample = tokenizer("How do I choose a learning rate?", padding="max_length", truncation=True,
                   max_length=MAX_LENGTH, return_tensors="pt")
with torch.inference_mode():
    logits = adapted(sample["input_ids"], sample["attention_mask"], sample["token_type_ids"])
raw = logits.softmax(-1)[0]
scaled = (logits/policy["temperature"]).softmax(-1)[0]
assert int(raw.argmax()) == int(scaled.argmax())
display(pd.DataFrame({"topic": LABELS, "raw": raw.numpy(), "temperature-scaled": scaled.numpy()}))'''),
md('''## 9. Self-check and further learning

1. Why does a pretrained encoder still need task labels?
2. What is the difference between freezing parameters and disabling dropout?
3. Can a different tokenizer be used if its vocabulary is the same size?
4. Why should a validation-derived confidence claim be tested on new inputs?

<details><summary>Worked answers</summary>

1. Its original objective did not define our six course categories. The supervised task must teach that mapping.
2. Freezing disables parameter-gradient updates. Evaluation mode changes dropout and other train/eval behavior. A frozen encoder can still behave randomly if dropout remains active.
3. No. Token-to-ID assignments and preprocessing must match the trained embedding rows, not just their count.
4. Validation was used to choose the policy. New representative inputs test whether its apparent reliability carries over to deployment.

</details>

**Further learning:** compare LoRA with full fine-tuning under a new predeclared protocol; evaluate consented real questions; measure seed sensitivity and confidence calibration on a fresh holdout. Preserve today’s results as the reference record.

**Next:** [Notebook 5 — From trained model to deployed app](05_deployment.ipynb).
''')]
finish('04_pretrained.ipynb',c)

c=opening('5','From trained model to deployed app',35,[
 'How do we turn an evaluated checkpoint into the same predictions in a real app?',
 'Define an inference contract and save all required artifacts.',
 'Export ONNX and compare it numerically against PyTorch.',
 'Separate model routing from source retrieval and handle uncertain input.',
 'Run, deploy, smoke-test and maintain the browser application.'
],'''Modules 7–8 ask whether a model can become a useful application and whether adaptation was worthwhile. Here we connect the saved model to a public app. The work includes preprocessing, source attribution, loading behavior, testing and an honest model card—not just a predict button.''')
c += [md('''## 1. The deployed product

Open **[Course Navigator](https://solomonpromise.github.io/course-navigator/)**. Enter a question, inspect the suggested topic, read the actual excerpts and follow a lesson link. Expand the details to see WordPiece tokens and the six model probabilities.

The app loads a fine-tuned encoder and head into the browser. Its inference uses **ONNX Runtime Web**, a runtime for an exported computation graph. A Web Worker keeps model computation off the page’s main interface thread. Inference does not need a hosted Python server, an API key or a paid inference provider.

The initial model/runtime download requires a connection. After assets are loaded, prediction operates locally; the app makes no request containing the question. External lesson links and repository/Colab links remain online resources. This is a scoped educational app, not a claim of comprehensive production certification.
'''),figure('01_text_pipeline.png','The inference pipeline with separate source retrieval.','Training updates parameters. Deployment applies the saved parameters and retrieves attributed course material.'),
md('''## 2. Save an inference contract, not just weights

The contract includes the exact model configuration, adapted weights, tokenizer vocabulary and normalization, maximum length, label order, pooling rule and confidence policy. A silent change to any one of these can produce wrong predictions without a shape error.

`artifacts/finetuned.pt` stores trusted tensor weights. `pretrained-source` supplies the encoder configuration and original checkpoint for reproducible loading. The browser only needs the adapted exported graph, tokenizer settings, policy and retrieval index. Hashes identify the exact shipped files.

Never load arbitrary uploaded pickle/checkpoint files. This project only loads its own fixed trusted artifacts with tensor-only loading. Learner input is plain text and never interpreted as HTML or code.
'''),code('''import hashlib
config = json.loads((ROOT / "app/config.json").read_text())
digest = hashlib.sha256((ROOT / "app/model.onnx").read_bytes()).hexdigest()
assert digest == config["model_sha256"]
print("Model SHA-256:", digest)
print("Label order:", config["labels"])
print("Input shape: (1,", config["max_length"], ") int64 for IDs, attention mask and token types")'''),
md('''## 3. Export the actual adapted computation

**ONNX** is a portable representation of a model’s operators and parameters. Export does not train a new model. We put the PyTorch model in evaluation mode and trace the fixed 96-token contract with opset 17. Batch size one matches the interactive app.

The reference model was exported by `scripts/evaluate_export.py`; the cell below can create a separate classroom export. The original file remains untouched. The exported model uses the trained encoder and our masked-mean topic head, not the unadapted base checkpoint.

Source: [ONNX Runtime Web deployment documentation](https://onnxruntime.ai/docs/tutorials/web/deploy.html).
'''),code('''model = load_reference("finetuned")
tokenizer = pretrained_tokenizer()
example = tokenizer("Why is attention useful?", padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt")
EXPORT_AGAIN = False
if EXPORT_AGAIN:
    output = ROOT / "artifacts/classroom"; output.mkdir(exist_ok=True)
    torch.onnx.export(model, (example["input_ids"], example["attention_mask"], example["token_type_ids"]),
        str(output / "model.onnx"), input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"], opset_version=17, dynamo=False)
print("Reference ONNX size (MB):", round((ROOT / "app/model.onnx").stat().st_size/1e6, 2))'''),
md('''## 4. Verify numerical parity before publishing

Two implementations need not be bit-for-bit identical, but their differences must be acceptably small. The reference check compares PyTorch and ONNX logits over 48 inputs: all authored probes, unrelated prompts and tokenizer edge cases. The maximum absolute logit difference is about **6.7 × 10⁻⁶**, below the fixed 10⁻⁴ tolerance.

Tokenizer parity is equally important. JavaScript WordPiece is checked against Hugging Face’s full 96-ID and attention-mask arrays, including accents, Chinese characters, whitespace, special tokens and a very long unknown word. Shape checks alone would miss a wrong word-ID mapping.
'''),code('''import onnxruntime as ort
session = ort.InferenceSession(str(ROOT / "app/model.onnx"), providers=["CPUExecutionProvider"])
fixtures = json.loads((ROOT / "tests/browser-fixtures.json").read_text())
largest_difference = 0.0
for fixture in fixtures:
    ids = np.array([fixture["ids"]], dtype=np.int64)
    mask = np.array([fixture["mask"]], dtype=np.int64)
    actual = session.run(None, {"input_ids": ids, "attention_mask": mask,
                               "token_type_ids": np.zeros_like(ids)})[0][0]
    largest_difference = max(largest_difference, float(np.abs(actual-fixture["logits"]).max()))
assert largest_difference < 1e-4
print("Inputs verified:", len(fixtures), "| Largest logit difference:", largest_difference)'''),
md('''## 5. Routing and retrieval have different jobs

The neural classifier estimates a broad topic. A **TF-IDF index** ranks actual sections by lexical similarity to the question, with a small 15% preference for the predicted topic. That preference is not another learned model. The index contains the complete public source library so it can return links and literal excerpts.

This is lexical retrieval, not neural semantic search or generated question answering. It can miss synonyms and favor repeated terms. We keep the distinction visible so learners know what each component contributes. A future semantic-retrieval experiment would need its own relevance labels and evaluation.
'''),code(source('coursenav/retrieval.py','LessonIndex')),
code('''index = LessonIndex()
question = "Why does my training loss fall while validation gets worse?"
batch = tokenizer(question, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
with torch.inference_mode():
    logits = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
    probabilities = (logits/config["policy"]["temperature"]).softmax(-1)[0]
topic = int(probabilities.argmax())
print("Predicted topic:", LABELS[topic], "| Scaled score:", round(float(probabilities.max()),3))
for section in index.search(question, topic):
    print(section["title"], section["url"], section["text"][:240], sep=" | ")'''),
md('''## 6. Make failure understandable

The app rejects blank text, caps input at 2,000 characters, and announces when the model truncates content to 94 WordPieces. Model loading is deliberate and reported. The first request downloads the model and runtime; later requests reuse the loaded session. A failed load produces a retryable message.

Low confidence or weak lexical overlap leads to **tentative suggestions**. This is a practical heuristic, not a guarantee against unrelated questions. Sources are always visible. Results are inserted as text nodes; input is never injected as HTML. Native controls, visible focus indicators, status announcements and phone layouts support access.

The runtime is pinned and served with the app, including its license/notices. It uses one WASM thread so GitHub Pages does not need cross-origin isolation headers. The browser’s worker and WebAssembly support remain deployment requirements.
'''),
md('''## 7. Run locally, then publish reproducibly

From the project folder:

```sh
python -m http.server 8820 --directory app
```

Open `http://localhost:8820`. A file opened directly with `file://` cannot reliably load a worker, model and data; use an HTTP server. No model training is needed to serve the app.

The GitHub repository contains a Pages workflow that uploads `app/` as the static site. The workflow also copies notebooks into the published downloads. The source, dataset, checkpoints, hashes and evaluation record remain reviewable. Publishing is separate from fitting; a notebook does not automatically push or change a public app.

**Release checklist:** run the split/model/tokenizer checks; execute all notebooks from clean kernels; verify local and public inference; inspect phone layout; check lesson links; record artifact hashes; then publish. `docs/verification.json` records the actual checks completed for this version.
'''),
md('''## 8. A classroom demonstration and a maintenance plan

Demonstrate one correct routing result, one ambiguous input and one unrelated request. Show the tokens, uncertainty note and source excerpt. Let learners try their own questions and explain why one successful example is not an evaluation.

For continued improvement, collect feedback with consent, define representative labeled questions, preserve a fresh holdout, compare against the lexical baseline, measure calibration and latency, and monitor broken source links. If the course structure changes, update source attribution and evaluate again. Retraining must produce a new versioned artifact and result record, not overwrite history silently.

The present reference run supports a teaching demo with useful lesson suggestions. It does not establish reliable behavior for arbitrary subjects, multilingual input, high-stakes decisions or actual learner traffic. The same engineering habit from Module 4 applies: describe what the evidence supports.

**Check yourself.** Why might an app return a useful source section even when the topic classifier is wrong? Why is a tokenizer checksum as important as a model checksum?

<details><summary>Worked answers</summary>

Retrieval uses the input’s lexical overlap and only a small topic preference; it can recover a relevant section despite a routing error. Conversely, a correct topic can still produce poor section rankings. The tokenizer defines which embedding rows the model sees; changed normalization or ID assignments invalidate the checkpoint’s meaning even when the model file is unchanged.

</details>

**You now have the full path:** real text → preprocessing → embeddings → recurrent and attention-based models → loss and parameter updates → evaluation → pretrained adaptation → export → a working application.
''')]
finish('05_deployment.ipynb',c)
