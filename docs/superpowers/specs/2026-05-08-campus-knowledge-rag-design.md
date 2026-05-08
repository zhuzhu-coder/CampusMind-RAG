# Campus Knowledge RAG Phase 1 Design

## Goal

Replace the recipe-specific RAG project with a campus knowledge base RAG that can ingest messy real-world campus documents and answer questions grounded in those documents.

This phase removes the `data/cook` recipe corpus, introduces `data/campus`, and updates the ingestion pipeline so the system no longer depends on clean Markdown recipe templates. The first release supports text-extractable `pdf`, `md`, and `txt` files only. Scanned PDFs are out of scope for now and will be reported as unsupported instead of failing the whole build.

## Scope

In scope:

- Retire the recipe dataset and recipe-specific language.
- Rebrand the system as a campus knowledge base.
- Support campus document categories such as regulations, teaching affairs, daily life notices, and announcements.
- Add robust parsing for non-standard document formatting.
- Support `pdf`, `md`, and `txt` inputs.
- Preserve retrieval strategies already in the project: vector, BM25, and hybrid RRF.
- Keep grounded answers and citations.
- Replace the old recipe eval default with a small campus smoke eval so the eval command still works after the rebrand.

Out of scope:

- OCR for scanned PDFs.
- FastAPI endpoints.
- Frontend UI.
- Incremental indexing.
- Authentication, permissions, and multi-user management.

## Target Data Layout

Replace the recipe corpus with a campus corpus:

```text
data/campus/
  regulations/
  teaching/
  life/
  notices/
  other/
```

Suggested starter files should be small but representative, covering:

- one or two regulations documents
- one teaching affairs notice
- one life/service notice
- one general announcement
- at least one text-extractable PDF

The corpus should be enough to run the system locally and verify the new parsing paths.

## Document Model

The old recipe fields are replaced by campus document metadata.

Required metadata fields:

- `doc_title`: human-readable document title
- `doc_category`: regulations / teaching affairs / life / notices / other
- `department`: issuing department when known
- `file_type`: `pdf`, `md`, or `txt`
- `source`: original file path
- `doc_id`: stable document identifier
- `chunk_id`: stable chunk identifier
- `chunk_index`: chunk order within the document
- `section`: title or inferred section label
- `page`: PDF page number when available
- `doc_type`: `parent` or `child`

Optional metadata fields:

- `source_name`
- `content_length`
- `line_count`
- `file_size`

Title inference priority:

1. Explicit title inside the document
2. First Markdown heading
3. First meaningful non-empty line
4. File stem
5. `未知文档`

## Ingestion and Parsing Design

The ingestion path must tolerate inconsistent formatting. It should not assume that every document has clean Markdown headings or a shared template.

### PDF

- Read text from each page.
- Preserve page number in metadata.
- If a page has no extractable text, log it and continue.
- If the entire PDF yields no extractable text, skip the document and mark it as unsupported.
- Do not attempt OCR in this phase.
- Skip or warn on fully unreadable PDFs.

### Markdown

- Prefer heading-aware splitting when headings exist.
- If headings are sparse or absent, fall back to generic paragraph/length splitting.
- Keep headings in chunk text when they help retrieval.

### TXT

- Normalize line breaks first.
- Split by blank lines when possible.
- Merge very short paragraphs.
- Fall back to fixed-length chunking when formatting is poor.

### Shared cleaning rules

- Remove repeated blank lines.
- Collapse stray whitespace.
- Preserve meaningful punctuation.
- Try to filter obvious page headers, footers, and page numbers when they are repeated.
- Never let one bad file stop the full indexing run.

## Chunking Strategy

The chunker should be document-type aware:

- Markdown: heading-aware first, fallback to generic chunking
- TXT: paragraph-aware first, fallback to fixed-size chunking
- PDF: page-aware first, then chunk within each page

Recommended defaults for Chinese campus documents:

- `chunk_size`: about 800 to 1200 Chinese characters
- `chunk_overlap`: about 100 to 200 Chinese characters

Chunking should favor sentence and paragraph boundaries when possible, but it must always have a deterministic fallback.

## Module Boundaries

To keep responsibilities clear, this phase should separate document loading from document preparation.

Likely file responsibilities:

- `code/rag_modules/data_preparation.py`: orchestrate loading, metadata enrichment, chunking, parent-child mapping, and statistics
- `code/rag_modules/document_ingestion.py`: file-type specific loading and text normalization
- `code/rag_modules/response_schema.py`: structured answer and source records
- `code/rag_modules/generation_integration.py`: grounded prompt and citation handling
- `code/main.py`: campus system entrypoint and response assembly
- `code/config.py`: default campus paths and environment overrides
- `code/evals/campus_smoke_eval_set.jsonl`: small campus smoke questions for keeping the eval command usable
- `code/evals/run_retrieval_eval.py`: point the default eval command at the campus smoke set

The key design rule is that file-format parsing should not live only inside the big orchestration module.

## Retrieval and Answer Flow

The retrieval stack stays the same:

- vector search
- BM25 search
- hybrid search with RRF

The answer path changes from recipe-oriented to campus-oriented:

1. receive a question
2. retrieve candidate chunks
3. deduplicate and align chunks to parent documents
4. build a grounded response with citations
5. return the final answer plus structured sources

The prompt must instruct the model to:

- answer only from retrieved campus documents
- avoid inventing policies, deadlines, locations, or contacts
- say clearly when the knowledge base does not contain enough evidence
- cite the source document, section, and page when available

## Error Handling

The system should fail soft, not hard:

- unreadable file: log warning and continue
- unsupported file type: skip with warning
- empty document: skip with warning
- PDF with no extractable text: mark as unsupported in this phase
- single-document parsing error: do not stop corpus indexing
- missing title: infer from filename or label as unknown

The build should still succeed if part of the corpus is bad, because campus uploads are often inconsistent.

## Testing and Verification

This phase should be verified with focused tests:

- ingest a mixed `data/campus` corpus
- confirm `pdf`, `md`, and `txt` all load
- confirm title inference works when headings are weak or missing
- confirm unsupported PDFs do not crash the run
- confirm response schema returns campus-style source records
- confirm the main CLI still initializes and answers from the new corpus
- confirm the eval command reads the campus smoke set instead of the retired recipe set

Recommended verification command:

```powershell
cd E:\RAG\code
python -m pytest tests -q -p no:cacheprovider
```

Additional manual checks:

- run the CLI once against the starter campus corpus
- inspect a sample answer to confirm citations reference the new metadata
- confirm `README.md` describes `data/campus`, not `data/cook`
- confirm the retired recipe corpus and recipe eval file are no longer the default project surface

## Rollout Notes

This phase is intentionally destructive with respect to the old recipe dataset. The new repository identity should be campus knowledge RAG, not recipe RAG. The recipe corpus and recipe-specific wording can be removed rather than preserved.

FastAPI and frontend work will come after this phase, once the ingestion layer is stable and the new corpus can be queried reliably.
