"""An explicit training loop with validation-only checkpoint selection.

Read this alongside Notebook 1. The same optimization/evaluation contract is
used by recurrence, our encoder and the pretrained encoder, making differences
in architecture visible without hiding the learning process inside Trainer.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from .data import ROOT, LABELS, MAX_LENGTH, TextDataset, load_records, fit_vocabulary
from .models import RecurrentClassifier, TinyTransformer, PretrainedClassifier

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"


@dataclass
class TrainConfig:
    """Fixed reference settings; test labels never influence these choices."""
    seed: int = 42
    batch_size: int = 32
    epochs: int = 20
    patience: int = 5
    learning_rate: float = .002
    weight_decay: float = .01
    max_grad_norm: float = 1.


def seed_everything(seed: int = 42):
    """Seed Python, NumPy and PyTorch; cross-machine equality is not promised."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def pretrained_tokenizer():
    """Use the exact bundled WordPiece vocabulary, avoiding network at inference."""
    from transformers import BertTokenizer
    vocabulary = {token: index for index, token in enumerate(
        (ROOT / "artifacts/pretrained-source/vocab.txt").read_text().splitlines())}
    tokenizer = BertTokenizer(vocab=vocabulary, do_lower_case=True)
    # Fail loudly if an API change silently created a five-special-token vocabulary.
    if tokenizer.vocab_size != 30522 or tokenizer.cls_token_id != 101:
        raise ValueError("Tokenizer does not match the pretrained BERT vocabulary")
    return tokenizer


def pretrained_dataset(rows, tokenizer=None):
    """Encode with BERT's own tokenizer; attention masks distinguish PAD from text."""
    tokenizer = tokenizer or pretrained_tokenizer()
    batch = tokenizer([row["text"] for row in rows], max_length=MAX_LENGTH,
                      padding="max_length", truncation=True, return_tensors="pt")
    return TensorDataset(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                         torch.tensor([row["label"] for row in rows], dtype=torch.long))


def make_model(name, vocabulary=None):
    """Construct an unadapted architecture; load a reference checkpoint separately."""
    if name in {"rnn", "lstm", "gru"}:
        return RecurrentClassifier(len(vocabulary), name.upper())
    if name == "transformer":
        return TinyTransformer(len(vocabulary))
    from transformers import BertModel
    encoder = BertModel.from_pretrained(ROOT / "artifacts/pretrained-source", attn_implementation="eager")
    # The classifier uses masked mean pooling, so the unused BERT pooler is removed.
    encoder.pooler = None
    model = PretrainedClassifier(encoder)
    if name == "frozen":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    elif name != "finetuned":
        raise ValueError(f"Unknown architecture {name}")
    return model


@torch.inference_mode()
def logits_for(model, dataset, batch_size=64):
    """Disable dropout/gradients and preserve row order during inference."""
    model.eval()
    return torch.cat([model(*batch[:-1]).cpu() for batch in DataLoader(dataset, batch_size=batch_size)])


def summarize(labels, logits):
    """Report accuracy, macro-F1 and per-class errors; no metric is fabricated."""
    predictions = np.asarray(logits).argmax(axis=1)
    return dict(accuracy=float(accuracy_score(labels, predictions)),
                macro_f1=float(f1_score(labels, predictions, labels=range(len(LABELS)), average="macro", zero_division=0)),
                confusion_matrix=confusion_matrix(labels, predictions, labels=range(len(LABELS))).tolist(),
                classification_report=classification_report(labels, predictions, labels=range(len(LABELS)),
                                                            target_names=LABELS, output_dict=True, zero_division=0))


def fit(model, train_data, validation_data, config: TrainConfig, frozen=False):
    """Train with AdamW; retain the checkpoint with the best validation macro-F1.

    Returns the restored best model, per-epoch measurements and elapsed seconds.
    This function accepts no test dataset: selection cannot accidentally use it.
    """
    generator = torch.Generator().manual_seed(config.seed)
    batches = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, generator=generator)
    counts = torch.bincount(train_data.tensors[-1] if hasattr(train_data, "tensors") else train_data.labels,
                            minlength=len(LABELS)).float()
    class_weights = len(train_data) / (len(LABELS) * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    best_score, best_epoch, best_state, history = -1., 0, None, []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        if frozen:
            model.encoder.eval()  # frozen features should not randomly change through dropout
        total_loss, examples = 0., 0
        for batch in batches:
            # 1 clear accumulated gradients; 2 forward; 3 loss; 4 backward; 5 update.
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-1])
            loss = criterion(logits, batch[-1])
            loss.backward()
            # Clipping limits an unstable update; it does not cure vanishing gradients.
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * len(batch[-1])
            examples += len(batch[-1])
        valid_logits = logits_for(model, validation_data)
        labels = validation_data.tensors[-1].numpy() if hasattr(validation_data, "tensors") else validation_data.labels.numpy()
        scores = summarize(labels, valid_logits.numpy())
        entry = dict(epoch=epoch, train_loss=total_loss/examples, validation_macro_f1=scores["macro_f1"],
                     validation_loss=float(nn.functional.cross_entropy(valid_logits, torch.tensor(labels))))
        history.append(entry)
        print(f"epoch={epoch:02d} loss={entry['train_loss']:.3f} val_f1={entry['validation_macro_f1']:.3f}", flush=True)
        if scores["macro_f1"] > best_score + 1e-5:
            best_score, best_epoch = scores["macro_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, time.perf_counter() - started


def load_reference(name):
    """Load trusted local tensor weights; do not deserialize unknown pickle files."""
    vocabulary = json.loads((ROOT / "artifacts/vocabulary.json").read_text())
    model = make_model(name, vocabulary)
    model.load_state_dict(torch.load(ROOT / f"artifacts/{name}.pt", map_location="cpu", weights_only=True))
    return model.eval()


def dataset_for(name, rows):
    """Select preprocessing that matches the architecture's saved weights."""
    if name in {"frozen", "finetuned"}:
        return pretrained_dataset(rows)
    return TextDataset(rows, json.loads((ROOT / "artifacts/vocabulary.json").read_text()))


def train_reference(name):
    """Run one predeclared experiment and persist its selected checkpoint/history."""
    seed_everything()
    train, valid = load_records("train"), load_records("validation")
    vocabulary = fit_vocabulary(train)
    (ROOT / "artifacts/vocabulary.json").write_text(json.dumps(vocabulary))
    config = TrainConfig()
    if name == "frozen":
        config.epochs, config.learning_rate = 16, .008
    if name == "finetuned":
        config.epochs, config.learning_rate = 12, .0002
    model = make_model(name, vocabulary)
    model, history, seconds = fit(model, dataset_for(name, train), dataset_for(name, valid), config, frozen=name=="frozen")
    torch.save(model.state_dict(), ROOT / f"artifacts/{name}.pt")
    result = dict(name=name, config=asdict(config), history=history, train_seconds=seconds,
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (ROOT / f"artifacts/{name}-run.json").write_text(json.dumps(result, indent=2))
    return result
