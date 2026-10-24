# Attribution and dependency notices

- **Course prose:** Solomon Promise, Modern Deep Learning & AI Engineering. Original public repository: https://github.com/solomonpromise/deep-learning-lab . Source snapshot/hashes are recorded in `data/manifest.json`. Course text is not relicensed by the dependencies below.
- **BERT-Tiny pretrained checkpoint:** Google, `google/bert_uncased_L-2_H-128_A-2`, revision `30b0a37ccaaa32f332884b96992754e246e48c5f`, Apache License 2.0. Citation: Turc, Chang, Lee and Toutanova, *Well-Read Students Learn Better: On the Importance of Pre-training Compact Models* (2019), https://arxiv.org/abs/1908.08962 . Adaptation adds a course-topic head and fine-tunes encoder weights; original pretraining-task head is excluded. The exported adapted model is a modified derivative. See `licenses/bert-Apache-2.0.txt` and the bundled model card.
- **ONNX Runtime Web:** Microsoft and contributors, 1.20.1, MIT. Exact distribution hashes are in `app/vendor/provenance.json`; original MIT license and third-party notices are retained in that folder.
- **Python libraries:** PyTorch, Hugging Face Transformers/tokenizers, NumPy, pandas, scikit-learn, Matplotlib, ONNX/ONNX Runtime and Jupyter retain their respective upstream licenses. Pinned reference requirements do not change those terms.
- **Typography:** optional Google Fonts delivery of DM Sans and Literata; system/Georgia fallbacks are used when unavailable. Fonts are not bundled or modified in this repository.

All project schematics and scientific figures were produced for this lab; plotted measurements are from the bundled reference runs. No stock figure or outside dataset is represented as an original course measurement.
