# Dataset: course prose and diagnostic questions

**Owner/source:** Solomon Promise's Modern Deep Learning & AI Engineering course, Modules 1–8. The source repository is `solomonpromise/deep-learning-lab`; `data/manifest.json` freezes commit `2928128b004b1113e044fb7ff1894d8b0b59ea54` and each notebook's SHA-256. The lab does not modify those source notebooks.

**Unit:** a non-overlapping prose window of up to 90 whitespace-delimited words, retained if the scratch tokenizer finds at least 25 tokens. The model then truncates to its 96-position tokenizer-specific contract. Six coarse labels are mapped from source module groups: Modules 1–2, 3–4, 5, 6, 7 and 8. Labels describe the source's instructional location; overlapping content can make them ambiguous.

**Extraction:** Markdown teaching prose from numbered sections; code/output/setup excluded; formatting, links, image references and mathematical markup cleaned. Source URLs remain in each record. This is existing educational text, not scraped user messages or invented reviews. Cleaning can omit mathematical detail, so the model sees a simplified text view rather than the complete curriculum.

**Split:** deterministic stratified split of complete source sections (seed 42, then 43). All windows from one source section stay in one split. Exact normalized duplicate passages are removed before splitting. This does not guarantee absence of semantic near-duplicates or overlapping topics across sections. Different sections of a lesson can appear in different splits. Use a lesson- or course-level holdout for broader generalization claims.

**Counts:** 1,752 passages; 337 sections; 1,201 train, 286 validation, 265 test. There are no missing labels. Training class weights use training counts only. Scratch vocabulary and TF-IDF classifier features are fitted on training only. Pretrained WordPiece is fixed independently of this dataset.

**Question probes:** 36 author-created English questions, six per topic, reviewed against the label definitions and frozen before training. Six unrelated prompts and additional Unicode/tokenizer edge cases support engineering checks. These are not actual learner traffic. None select checkpoints, temperature or confidence threshold. Their small size and constructed wording limit inference about real-world quality.

**Retrieval:** `data/sections.json` is a full public reference library. Indexing all its sections permits the app to retrieve source material; it does not add those sections' labels to classifier training. No labeled relevance benchmark was collected, so classification scores must not be described as retrieval accuracy.

**Limits:** English course-specific prose, coarse module-derived labels, correlated sections, short inputs, no representative user data, no multilingual evaluation, no comprehensive out-of-domain benchmark. A new course version or user population requires new evaluation.

**Attribution and rights:** course prose is credited to Solomon Promise. Public availability is not an additional license grant; third-party model/runtime licenses do not license the course material. Learners may use the lab under the course's applicable teaching permissions.
