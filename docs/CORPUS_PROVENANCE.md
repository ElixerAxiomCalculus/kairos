# Corpus provenance

KAIROS keeps retrievable passages separate from evaluation labels and records a manifest beside every imported corpus. Imports are bounded, deterministic for a fixed source response, seed, limit, and revision, and use namespaced document IDs to prevent collisions.

## Included profiles

| Profile | Paths | Documents | Chunks | Intended use |
|---|---|---:|---:|---|
| `sample` | `corpus/sample` | 8 | 37 | Project-authored fictional policies and event operations |
| `science` | `corpus/imported/scifact` | 12 | 29 | Bounded scientific-abstract retrieval demonstration |
| `general` | `corpus/imported/squad` | 8 | 16 | Bounded general reading-comprehension retrieval demonstration |
| `multi_domain_demo` | all three paths above | 28 | 82 | Routing, rejection, and multi-domain stress testing |

Counts are those measured by `python -m kairos semantic-bench` in run `20261018T095654Z`. The default remains the 37-chunk `sample` profile so the ordinary laptop demonstration stays lightweight.

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

This import contains corpus passages only. It does not claim a SciFact task evaluation because claim/qrels splits were not imported for this bounded demonstration.

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

## Academic citations

- David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. “Fact or Fiction: Verifying Scientific Claims.” *Proceedings of EMNLP 2020*, pages 7534–7550. DOI: [10.18653/v1/2020.emnlp-main.609](https://doi.org/10.18653/v1/2020.emnlp-main.609).
- Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.” *NeurIPS 2021 Datasets and Benchmarks Track*. [OpenReview](https://openreview.net/forum?id=wCu6T5xFjeJ).
- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. “SQuAD: 100,000+ Questions for Machine Comprehension of Text.” *Proceedings of EMNLP 2016*, pages 2383–2392. DOI: [10.18653/v1/D16-1264](https://doi.org/10.18653/v1/D16-1264).

## Reproducible imports

Online imports use the Hugging Face Dataset Viewer API through the Python standard library. Pin the revision and validate the downloaded response checksum:

```bash
python -m kairos import-corpus --dataset scifact --destination corpus/imported/scifact --revision b3b5335604bf5ee3c4447671af975ea25143d4f5 --limit 12 --seed 42 --expected-checksum ebd68a151093e62b046312d869cd84d90b63e7d27cb8c11cc672f780767b7fb6
python -m kairos import-corpus --dataset squad --destination corpus/imported/squad --revision 7b6d24c440a36b6815f21b70d25016731768db1f --limit 12 --seed 42 --expected-checksum 4e29bcb762d254cf9b8aa7f255de952197c9cc0d523a5461372226053629d3c0
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
