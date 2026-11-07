# docmem

Index local documents into **cognee on nanolance**, keeping only vectors and provenance, never the text.
Questions are answered by reading the cited byte ranges back from the files. Your OpenAI-compatible
server writes the answers.

```
docmem scan  corpus/  ──JSON lines──▶  docmem index --db kb/      (cognee: Ladybug graph + Lance vectors)
                                              ▲
docmem ask / docmem eval  ── vector search ───┘ ── file:// offset+length ──▶ text ──▶ your chat model
```

## Setup

```sh
uv venv && source .venv/bin/activate
uv pip install -e ../.. -e <nanolance>/bindings/python -e .      # ../.. is this cognee fork
```

The two cognee changes the app needs are in this fork, on this branch (see *What changed in cognee*).

## Use

```sh
docmem scan ~/docs | docmem index --db ~/kb          # first time, and after any edit (only changes are re-embedded)
docmem scan ~/docs | docmem index --db ~/kb --prune  # also forget files deleted from ~/docs

export DOCMEM_API_BASE=https://your-server/v1 DOCMEM_API_KEY=... DOCMEM_MODEL=your-model
docmem ask --db ~/kb "how is the retry budget computed?"
docmem eval --db ~/kb --questions my_questions.jsonl --judge --out report/
docmem stats --db ~/kb
```

- **Key handling:** `--api-base`, `--api-key` and `--model` can replace the environment variables. The key is never written to disk.
- **Embeddings:** local `fastembed:BAAI/bge-small-en-v1.5` by default, no key needed. To embed on your server, create the
  index with `--embed openai:text-embedding-3-small --embed-dims 1536`. `kb/docmem.json` pins the model, since
  questions must be embedded with the same one.
- **ask options:**
  - `-k 6`: passages given to the model.
  - `--context section`: give the whole enclosing section, up to `--max-bytes`.
  - `--rerank 20`: fetch 20 candidates and let the chat model keep the best k.
  - `--no-answer`: retrieval only, no API.
- **Changed files:** every passage's SHA-256 is checked when it is read back. A file changed since indexing shows
  `[CHANGED: re-run scan | index]`, and that passage is not given to the model.

### Multi-hop

A question that needs facts from two or three articles often retrieves only one of them, because one question
embedding lands near one article. With `--hops N` the model may ask for more: if the excerpts lack a fact, it
replies `SEARCH: <query>` lines, `ask` retrieves `--hop-k` passages (default 3) for each query, adds the new ones
(no repeats) and asks again. After N rounds, or once the model answers, it answers from everything gathered, capped
at `--max-passages` (default 10). The number of answer calls is at most N+1. A question answered from the first
passages costs one call, as before. `ask` prints the searches it made. `eval --hops N` also reports the
searches and calls per question, and `llm_calls_per_question`. The default `--hops 0` is unchanged.

## The stream (protocol 1)

One JSON object per line; per document:

```json
{"type":"doc","uri":"file:///…/a.md","sha256":"…","size":696658,"mtime":…,"title":"Afghanistan"}
{"type":"section","uri":"…","path":["Afghanistan","Economy"],"level":2,"offset":81234,"length":5120,"line":290}
{"type":"passage","uri":"…","path":[…],"offset":81260,"length":912,"line":292,"end_line":294,"sha256":"…","text":"…"}
{"type":"end","uri":"…","sections":384,"passages":849}
```

then `{"type":"scan_end","roots":[…],"uris":[…]}`.
- **Byte ranges:** offsets and lengths count bytes of the file as scanned.
- **Text:** a passage carries its `text` for the indexer to embed; the indexer does not keep it.
- **Other scanners:** any program that writes this stream can feed the indexer: a web-docs crawler, a PDF scanner. For
  them `uri` may be `https://…`. The `file://` reader in `ask` is the only part that would need a sibling.

**Cutting:** documents split into heading-path sections. A section's blocks (paragraphs, lists, tables, code fences)
are merged up to `--target-words` (180). A block over `--max-words` (320) is split at sentence ends. A passage never
crosses a heading. Each passage is embedded as `"Doc › Chapter › Section\n\n" + text`.

## What is stored

Per file:
- a `DocFile` node: URI, SHA-256, size, title;
- a `DocSection` node per heading: path, byte range, line;
- a `DocPassage` node per passage: byte range, lines, SHA-256, its section's range.

Edges run passage → section → file. Vectors: one per passage and one per heading path. **No passage text** is
in the graph or the vector table. The test below greps the database for passage sentences and finds none.

## Updates

A passage's identity is (file, SHA-256 of its bytes, occurrence). On a re-run:
- An unchanged file (same hash) is skipped.
- In a changed file, only new passages are embedded.
- Surviving passages get their new offsets, written graph-only with no embedding.
- Vanished passages are deleted from the graph and the vector tables.

## Measured (this sandbox: 4 slow CPU cores, no GPU)

Test corpus: `tools/make_wiki_corpus.py` builds 6 markdown files of ~650 KB each (4.0 MB in all). Each holds 8 long
Wikipedia articles, with 2,359 headings in total. `examples/wiki_questions.jsonl` has 30 questions: 22 single-fact,
5 that need two sections, and 3 the corpus cannot answer.

| step | result |
|---|---|
| scan, 6 files | 4,898 passages + 2,359 sections in **1.2 s**; every offset re-read and hash-checked |
| index, local bge-small | **354 s**, 14 passages/s; nearly all of it CPU embedding |
| re-run, nothing changed | 1.2 s of work (plus ~11 s to import cognee) |
| edit near the top of a 700 KB file (shifts every offset) | 1 passage embedded, 848 kept with new offsets, 1 deleted: **4.4 s** |
| `--prune` after deleting a file | 1 file, 440 sections and 798 passages removed: 3.1 s |
| retrieval, 27 answerable questions | hit@1 **0.96**, hit@3 **1.00**, MRR 0.98; 34 ms median per question (warm) |
| index size | Ladybug graph 23 MB, Lance vectors 16 MB, for 4.0 MB of text |

- **Answer quality:** not measured yet. It needs your server: run `docmem eval … --judge`.
- **Mock server:** `tools/mock_openai.py` is a word-overlap stand-in for an OpenAI-compatible server. Against it the chat, rerank,
  judge and remote-embedding paths all ran; its scores mean nothing.
- **Index size:** the index is larger than the text it points to. Most of the graph is cognee's per-node metadata. The
  vectors are float32: 384 × 4 bytes × 7,263.

## Keyword index

`docmem index` also builds a BM25 keyword index beside the vectors, in `<db>/system/databases/docmem_keywords.lancedb`
(nanolance tables; `--no-keywords` skips it). The indexer reads each new passage's text once: its terms and counts go
into postings, and the text is dropped.

```sh
docmem kwsearch --db ~/kb "Riverside College Widnes Ofsted"   # top 10: score, passage id, heading path, uri, line
```

- **Tokens:** lowercase NFKC runs of letters and digits; internal `-`, `'` and `.` are kept (`REQ-1234`, `o'neill`,
  `3.5`); a trailing `'s` is dropped. No stemming, no stop words.
- **Tables:** `kw_passages` (num, passage_id, length, deleted) and `kw_postings` (term, segment, nums, tfs), one postings
  segment per flush. A BTree on `term` makes a search read only its terms' rows.
- **Deletes:** a deleted passage is marked in `kw_passages` and drops out of results at once; its postings are removed
  by compaction.
- **Size:** `python tools/kw_size.py --db ~/kb` compares the keyword tables with the text they cover, and estimates a
  delta and bit-packed layout.
- **Existing indexes:** an index built with `--no-keywords` has no keyword rows for its passages. Index into a new `--db`
  to add them.

## Concepts from an ontology

`docmem ontology load` stores a concept scheme in the index: IPTC Media Topics JSON (retired concepts skipped) or any
SKOS RDF file. Each concept is embedded from its label and definition, and has a `broader` edge to its parent.
`docmem classify` then links each document (and, with `--level both`, each section) to its nearest concepts by the
cosine similarity between its centroid (the normalized mean of its stored passage vectors) and the concept vectors.
Re-running replaces the scheme's `about_concept` edges. Nothing calls an LLM and no source file is read.

```sh
docmem ontology load --db ~/kb --skos mediatopic.json --scheme iptc
docmem classify --db ~/kb --scheme iptc --top 3
docmem concepts --db ~/kb --doc "Some Article"    # scores and ancestor chains
docmem concepts --db ~/kb --tree                  # top two levels, with document counts
docmem ask --db ~/kb --concepts 3 "question"      # add passages of documents sharing a level-2+ concept
```

- `--query-prefix` on `ontology load` embeds the concepts with the bge query instruction. It replaces the stored
  concept vectors for that scheme, so re-load without it to switch back.
- `--concepts N` (on `ask` and `eval`) follows only concepts at level 2 or deeper; broad ones like "sport" are not
  followed.

**Pivot retrieval** (`--pivot N`, on `ask` and `eval`, no LLM) retrieves through a shared idea instead of outward
from the top passages. The concepts the question names (a label occurs in it as whole words, e.g. "viscosity") are
taken first, then the `--pivot-concepts C` (default 2) concepts nearest the question's vector. Passages of the
documents or sections about those concepts are ranked by similarity, and the best passage of each of up to N
documents is added. `--pivot-scheme` limits it to one scheme; `eval` reports the chosen concepts per question as
`pivot_concepts`.

## Wikipedia samples by topic and their category concepts

`docmem scan-wiki --seed-pages FILE` samples the articles that seed pages link to, not random articles. FILE has one
exact index title per line, optionally `title|field` (`examples/engineering_seeds.txt`). Each seed's article links are
the candidates; a redirect is followed once; disambiguation, list / outline / glossary / index pages and pages under
300 words are left out. The sample is stratified by field (equal shares, rounded; a small field's leftover goes to the
others), each article is drawn once, and every doc record carries `"field"`. Without `--seed-pages` the sample is random,
as before. Pool sizes, per-field counts and downloaded MB go to stderr.

`docmem wiki-categories --db DB --cache DIR --out FILE.ttl --depth 2` turns the Wikipedia categories of the indexed
articles into a SKOS concept scheme: each category page is read from the dump, its parents up to `--depth` levels are
added, maintenance categories are dropped and cycles are broken (the closing edge is dropped and counted). Load it with
`docmem ontology load --skos FILE.ttl --scheme NAME`. `FILE.doc_categories.json` (next to the Turtle file) maps each
indexed title to its kept categories, for evaluation. Nothing calls an LLM; category texts are not written to disk.

## What changed in cognee (this fork)

1. **`DataPoint.metadata["store_index_text"] = False`** (LanceDB adapter): the index field is embedded, and the vector
   row is stored with an empty `text`.
2. **`Field(exclude=True)`** now survives `copy_model`. The graph-node copy can still be read for embedding, but
   `model_dump()` leaves the field out, so graph adapters do not store it. Test:
   `cognee/tests/unit/modules/storage/test_copy_model_exclude.py`.

Cognee's own retrievers (`CHUNKS`, `RAG_COMPLETION`, …) expect stored chunk text. So `ask` does its own retrieval over
cognee's stores: a vector search, then the graph nodes, then a read of the file.
