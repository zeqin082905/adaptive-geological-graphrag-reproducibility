# Adaptive Geological GraphRAG: Reproducibility Package

This repository accompanies the manuscript **“An Adaptive Dual-Pathway
Retrieval-Augmented Generation Method for Intelligent Question Answering over Geological
Survey Archives”**, submitted to *Engineering Applications of Artificial Intelligence*.

It provides the method implementation, experiment drivers, configuration parameters,
evaluation scripts, aggregate results reported in the manuscript, and a fully synthetic
demonstration dataset. No restricted geological report, expert-authored answer, retrieved
passage, or institutional archive is included.

## What can be reproduced

- Document loading and cleaning for PDF, DOCX, and text inputs.
- Geological text chunking with overlapping windows and cross-page bridging.
- Vector retrieval and GraphRAG integration.
- Entity normalization, semantic-dispersion routing, query decomposition, reranking,
  graph-text evidence fusion, and provenance-constrained answer generation.
- G0--G3 ablation configurations and automated evaluation workflow.
- An end-to-end functional demonstration using the synthetic files in `examples/`.

The exact numerical results in the paper depend on restricted third-party geological
archives and an expert-validated question-answer set that the authors are not authorized
to redistribute. See [Data availability](docs/DATA_AVAILABILITY.md).

## Repository layout

```text
config/                         Runtime parameters and ablation definitions
src/                            Core implementation
scripts/                        Index conversion and evaluation utilities
examples/synthetic_corpus/      Fictional documents released for functional testing
examples/synthetic_qa.json      Fictional evaluation examples
results/aggregate_metrics.csv   Aggregate values reported in the manuscript
docs/                           Reproduction and data-restriction documentation
```

## Environment

The experiments were designed for Python 3.10 or later and a local
[Ollama](https://ollama.com/) endpoint. Install the dependencies in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pull the models used by the public example:

```bash
ollama pull qwen2.5:14b
ollama pull qwen2.5:32b
ollama pull nomic-embed-text
```

Model names and endpoints can be overridden with the variables documented in
`.env.example`. Never commit real credentials.

## Synthetic demonstration

The example documents are fictional and do not reproduce or paraphrase any restricted
archive. Build the public demonstration indexes with:

```bash
python main.py build \
  --corpus1-dir examples/synthetic_corpus/authoritative \
  --corpus2-dir examples/synthetic_corpus/supplementary
```

Then run a query:

```bash
python main.py query "What controls mineralization in the North Ridge district?" --json
```

GraphRAG indexing invokes a local language model and can be computationally expensive.
For a quick text-path check, append `--skip-graphrag` to the build command and query with
`--mode naive_rag`.

## Paper experiment configuration

The four configurations are documented in `config/ablation_modes.json`:

- **G0:** naive vector RAG.
- **G1:** graph retrieval without adaptive decomposition.
- **G2:** vector retrieval with query decomposition.
- **G3:** complete adaptive dual-pathway system.

The public scripts accept a user-provided dataset with the schema in
`docs/DATA_SCHEMA.md`. They do not silently download or embed the restricted study data.

## Reported aggregate results

`results/aggregate_metrics.csv` contains only manuscript-level aggregates. It contains no
question, answer, source passage, document identifier, or expert annotation.

## Citation

Please use the metadata in `CITATION.cff`. A versioned archival DOI can be added after the
repository is deposited in Zenodo.

## License

Code is released under the MIT License. The synthetic demonstration data are also covered
by this repository license. This license does not apply to the restricted source archives,
which are not distributed here.

