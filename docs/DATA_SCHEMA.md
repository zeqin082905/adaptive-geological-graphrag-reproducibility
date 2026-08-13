# Input and evaluation schemas

## Document corpora

The loader accepts `.pdf`, `.docx`, and UTF-8 `.txt` files. Two physically separate input
directories are expected:

- `corpus1`: authoritative documents used for knowledge-graph construction.
- `corpus2`: supplementary documents used for text-vector retrieval.

Files may contain ordinary prose. Internal document identifiers are generated from file
names by the loader; users should replace sensitive file names before using the code.

## Question-answer JSON

The public evaluation utilities expect a JSON array. Each item uses the following minimal
schema:

```json
{
  "id": "SYN-Q01",
  "question": "A question in natural language",
  "reference_answer": "A reference answer",
  "relevant_document_ids": ["synthetic_report_01"]
}
```

Some evaluation scripts also accept optional fields such as `category`, `keywords`, or
`supporting_facts`. See `examples/synthetic_qa.json` for non-sensitive examples.

## Generated results

Generated result files may include full answers and retrieved passages. Treat them as
sensitive whenever private input documents are used. Do not commit them to a public
repository. The default `.gitignore` excludes common output and index locations.

