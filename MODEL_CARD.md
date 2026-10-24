# Course Navigator topic classifier

## Intended use

An educational lesson-finding demonstration for English questions/passages about Modules 1–8 of this course. It predicts six topics and uses a separate lexical index to retrieve attributed sections. It is not a generative tutor, a universal topic classifier or a validated high-stakes decision system.

## Model and adaptation

Base: [Google BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2), revision `30b0a37ccaaa32f332884b96992754e246e48c5f`, Apache-2.0. Two layers, two heads, width 128, 30,522 WordPieces. A masked-mean representation feeds a six-class head. The pretraining language-model head/pooler are not used. Full fine-tuning uses the real course training split with class-weighted cross-entropy and AdamW; exact configuration, epochs, parameter counts and timing are in `artifacts/finetuned-run.json`.

Checkpoint selection uses validation macro-F1 only. The exported ONNX graph is the selected adapted encoder plus task head. A single input has three int64 tensors `(1,96)`. The application uses at most 94 content WordPieces plus CLS/SEP. No quantization is applied. Python weights are loaded with `weights_only=True` from trusted bundled files.

## Evaluation and deployment decision

Held-out passage macro-F1: **0.711**; authored diagnostic question accuracy: **24/36**. The lexical baseline reaches **0.773** and **24/36**. The neural model is deployed to teach the complete pretrained adaptation/export workflow, not because it wins the overall baseline comparison. The app and notebooks disclose this decision.

One seed and course-internal section splits do not establish a reliable ranking for other settings. Training/test distributions differ from real user questions. Labels overlap. Refer to `artifacts/results.json` for every model's metrics and class-specific errors.

## Confidence handling

Validation-only temperature scaling uses T≈2.364. A threshold≈0.625 selects a validation subset with ≈81.2% accuracy and ≈74.5% coverage. These are policy-selection measurements, not independent guarantees. A weak lexical-overlap check also marks tentative suggestions. Neither mechanism is a validated detector of unrelated input. Confident mistakes remain possible.

## Delivery and privacy

ONNX Runtime Web 1.20.1, one WASM thread, fixed input shape, browser worker. Model/runtime initially download to the device; question text is not sent to an inference API. The site does not persist questions, collect analytics or use an inference credential. GitHub Pages, external font delivery and outgoing resource links still make ordinary network requests. The app is not promised to work fully offline.

## Maintenance

Version model/tokenizer/label map/policy/index together. Verify numerical parity and artifact hashes after changes. Collect representative consented question/relevance labels before product-quality claims. Keep new evaluation sets separate from iterative development. Preserve prior result records.
