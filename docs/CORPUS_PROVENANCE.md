# Corpus provenance

KAIROS keeps retrievable passages separate from evaluation labels and records a manifest beside every imported corpus. Imports are bounded, deterministic for a fixed source response, seed, limit, and revision, and use namespaced document IDs to prevent collisions.

## Included profiles

| Profile | Paths | Documents | Chunks | Intended use |
|---|---|---:|---:|---|
| `sample` | `corpus/sample` | 8 | 37 | Project-authored fictional policies and event operations |
| `science` | `corpus/imported/scifact` | 12 | 29 | Bounded scientific-abstract retrieval demonstration |
| `general` | `corpus/imported/squad` | 8 | 16 | Bounded general reading-comprehension retrieval demonstration |
| `general_expanded` | `corpus/imported/squad_v2_1` | 21 | 41 | Expanded external-label retrieval evaluation |
| `multi_domain_demo` | all three paths above | 28 | 82 | Routing, rejection, and multi-domain stress testing |
| `multi_domain_v2_1` | sample + SciFact + expanded SQuAD | 41 | 107 | V2.1 multi-domain evaluation profile |

Counts are those measured for V2.1 on 2026-10-18. The default remains the 37-chunk `sample` profile so the ordinary laptop demonstration stays lightweight.

## SciFact subset

- Upstream dataset: [BeIR/scifact on Hugging Face](https://huggingface.co/datasets/BeIR/scifact)
- Original work: [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/)
- BEIR packaging/benchmark: [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://openreview.net/forum?id=wCu6T5xFjeJ)
- License reported by the dataset card/API: CC BY-SA 4.0
- Dataset revision: `b3b5335604bf5ee3c4447671af975ea25143d4f5`
- Configuration/split: `corpus` / `corpus`
- Import: 12 documents sampled with seed 42 from a 48-row bounded response
- Source-response SHA-256: `ebd68a151093e62b046312d869cd84d90b63e7d27cb8c11cc672f780767b7fb6`
- Output: 12 Markdown documents, 29 index chunks
- ID mapping: source corpus ID `118568` becomes document `SCIFACT:118568`; chunks use IDs such as `SCIFACT:118568:C01`

Official evaluation labels are imported separately from
[BeIR/scifact-qrels](https://huggingface.co/datasets/BeIR/scifact-qrels):

- Qrels revision: `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029`, configuration `default`, split `test`.
- Paginated qrels-response aggregate SHA-256: `5c8934df80c4adbf1f53eb3d7bfb595106c6306f30042f8cecb223ddb98243a7`.
- Paginated query-response aggregate SHA-256: `c077a1bba64a155ec2e6e9d4f04a7e7f0bd5d0eebba4c1b5c4549b5e8b802c07`.
- Joined label-file SHA-256: `f6a6e4f8ebf17c372df8883472bc6bd7f76cf6f58525cd4f18007dd1d383b23b`.
- Exactly one official test qrel intersects the committed 12-document corpus subset. Queries and qrels are never indexed.

The measured 1/1 retrieval result is therefore a pipeline/provenance check, not a representative SciFact benchmark score.

## SQuAD subset

- Upstream dataset: [SQuAD on Hugging Face](https://huggingface.co/datasets/rajpurkar/squad)
- Original work: [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/)
- License reported by the dataset card/API: CC BY-SA 4.0
- Dataset revision: `7b6d24c440a36b6815f21b70d25016731768db1f`
- Configuration/split: `plain_text` / `train`
- Import: 12 question rows sampled with seed 42 from a 48-row bounded response, deduplicated to 8 context documents
- Source-response SHA-256: `4e29bcb762d254cf9b8aa7f255de952197c9cc0d523a5461372226053629d3c0`
- Output: 8 Markdown documents, 16 index chunks, 12 records in `evaluation-labels.jsonl`
- ID mapping: each normalized context is content-addressed, for example `SQUAD:3787a46bd23562af`; chunks append `:C01`

Questions, answer text, and answer offsets are stored only in `evaluation-labels.jsonl`. They are not placed in retrievable Markdown and therefore cannot leak answer labels into the index.

### Expanded V2.1 SQuAD evaluation profile

- Same upstream revision, configuration, split, license, transformation, and ID scheme as the original profile.
- Import: 48 question rows selected with seed 42 from a 100-row bounded response, deduplicated to 21 context documents.
- Source-response SHA-256: `c5ad3e6f4e4ab3fb83bd79999e8937c9bb3557e277e1e6aa41243ec96734424f`.
- Output: 21 Markdown documents, 41 index chunks, and 48 non-indexed records in `evaluation-labels.jsonl`.
- This profile augments rather than overwrites the original small demonstration corpus.

### PSRC V2.2 controller-dataset use

`benchmark/datasets/controller_v2_2/` reuses the non-indexed SQuAD questions/document labels and the one
intersecting SciFact query/qrel as independently sourced controller examples. It does not copy answer text into
the retrieval corpus. Source documents/question families are kept wholly within one train, calibration, or
external split; individual prefixes are never randomly split.

These are written QA questions. V2.2's deterministic cumulative segmentation and timestamps are labeled
`synthetic written-question segmentation; not recorded speech` in every record. Their use does not convert them
into a natural spoken-dialogue dataset. The controller dataset manifest records upstream dataset name, source
question ID, source revision, corpus profile, annotation source, timing source, split, and content hash.

The same CC BY-SA 4.0 attribution/share-alike obligations and the academic citations below apply to these
derived controller examples. Project-authored KAIROS streams remain project content and are identified separately.

### IC-COS V2.3 controller-dataset use

`benchmark/datasets/controller_v2_3/` is append-only over V2.2. It preserves the imported SQuAD/SciFact source
IDs, immutable revisions, labels-outside-index boundary, licenses, and citations above. It adds candidate-lineage
intent annotations, canonical online features, counterfactual action labels, separate train/calibration/development
groups, and a frozen final manifest. The former V2.2 held-out split is named `historical_frozen` because its
published results already informed engineering decisions.

The 12 new final questions are project-authored, use `corpus/sample`, and are excluded from V2.3 training,
calibration, development selection, and threshold selection. They are not independently authored and do not add
third-party corpus content or a new license. The manifest records `independent_authorship: false`; this is limited
final validation, not external generalization evidence. External profile evaluation continues to use the CC
BY-SA 4.0 SQuAD and SciFact subsets under their existing attribution/share-alike terms.

## Academic citations

- David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. “Fact or Fiction: Verifying Scientific Claims.” *Proceedings of EMNLP 2020*, pages 7534–7550. DOI: [10.18653/v1/2020.emnlp-main.609](https://doi.org/10.18653/v1/2020.emnlp-main.609).
- Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.” *NeurIPS 2021 Datasets and Benchmarks Track*. [OpenReview](https://openreview.net/forum?id=wCu6T5xFjeJ).
- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. “SQuAD: 100,000+ Questions for Machine Comprehension of Text.” *Proceedings of EMNLP 2016*, pages 2383–2392. DOI: [10.18653/v1/D16-1264](https://doi.org/10.18653/v1/D16-1264).

## Reproducible imports

Online imports use the Hugging Face Dataset Viewer API through the Python standard library. Pin the revision and validate the downloaded response checksum:

```bash
python -m kairos import-corpus --dataset scifact --destination corpus/imported/scifact --revision b3b5335604bf5ee3c4447671af975ea25143d4f5 --limit 12 --seed 42 --expected-checksum ebd68a151093e62b046312d869cd84d90b63e7d27cb8c11cc672f780767b7fb6
python -m kairos import-corpus --dataset squad --destination corpus/imported/squad --revision 7b6d24c440a36b6815f21b70d25016731768db1f --limit 12 --seed 42 --expected-checksum 4e29bcb762d254cf9b8aa7f255de952197c9cc0d523a5461372226053629d3c0
python -m kairos import-scifact-eval --destination corpus/imported/scifact --corpus-revision b3b5335604bf5ee3c4447671af975ea25143d4f5 --qrels-revision 2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029
```

Useful safety modes:

```bash
python -m kairos import-corpus --dataset scifact --destination corpus/imported/scifact --dry-run
python -m kairos import-corpus --dataset squad --destination corpus/imported/squad --validate-only
```

Adapter tests use `tests/fixtures/datasets/` and require no network. Each generated manifest records upstream URL, revision, split, license, source checksum, per-document source ID, and per-document content checksum.

## Attribution, privacy, and use boundaries

The imported material remains subject to CC BY-SA 4.0 attribution and share-alike obligations; downstream distributors should review the upstream dataset cards and original sources before redistribution. Dataset metadata is not a legal opinion, and the upstream records may contain additional source-specific rights or privacy considerations.

SciFact abstracts can support retrieval demonstrations and research summaries, not personalized medical advice. SQuAD is a static reading-comprehension corpus, not a source for current prices, availability, weather, hospital proximity, property listings, or financial rates. KAIROS must abstain when a question requires missing, personalized, or live data.

FiQA, HotpotQA, and RAGBench were not imported in this implementation. Adding them remains conditional on a separate license, contamination, memory, and task-fit review.
