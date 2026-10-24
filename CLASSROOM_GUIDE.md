# A 3-hour-30-minute teaching route

Prepare the complete project, dependencies, Jupyter kernel and public app before class. Open all five notebooks and execute their setup cells once. Keep `RUN_TRAINING=False` for the guided route; run one minibatch update and optionally one short training demonstration. The supplied checkpoints allow you to teach the process without waiting for every model.

| Time | Activity |
|---|---|
| 00:00–00:10 | Show the app; define the task and the expected output. |
| 00:10–00:35 | Notebook 1: real text, split, vocabulary, embeddings, RNN, loss/backward/update. |
| 00:35–00:55 | Notebook 2: manual gate calculations; compare recurrent behavior and errors. |
| 00:55–01:45 | Notebook 3: attention arithmetic, masks, block implementation and inspection. |
| 01:45–02:40 | Notebook 4: pretrained tokenizer, frozen vs tuned encoder, evaluation and LoRA concept. |
| 02:40–03:15 | Notebook 5: artifacts, ONNX parity, source retrieval and deployed application. |
| 03:15–03:30 | Learner questions, three app trials and a recap of the full pipeline. |

## Keep three promises visible

1. Every score is a recorded measurement with a known split. No model is made to win for storytelling.
2. Training code is visible, but lengthy/repeated runs use prepared checkpoints. Independent study includes the full loop and extension experiments.
3. A deployable model needs preprocessing, evaluation, input handling and an explicit inference contract.

## Useful live questions

- What does a token ID mean before the embedding table is trained?
- Have weights changed immediately after `backward()`?
- Why do gates near one not all mean the same thing in every published equation?
- Which tokens can this encoder read, and how would causal generation differ?
- What did the pretrained model know before seeing our topic labels?
- Why is TF-IDF still worth deploying when a neural alternative exists?
- Can correct topic classification still return a poor source section?
- What evidence would we need before calling this app reliable for actual learner questions?

## Fallbacks

If training is slow, load the checkpoint and trace one batch. If a learner cannot install dependencies, use Colab after preparing the requirements. If browser model loading fails, use Notebook 5's ONNX example or Notebook 4's PyTorch inference; the source excerpts and saved results remain available. Use the public app in a tested modern browser and verify it before class.

Future study: repeated seeds, a fresh lesson-level holdout, consented real-question labels, semantic retrieval with relevance evaluation, a trained LoRA comparison, or a separate causal text-generation project. These are extensions, not silently completed experiments.
