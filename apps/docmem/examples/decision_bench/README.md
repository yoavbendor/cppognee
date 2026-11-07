# Decision benchmark item sets

Frozen item sets for `tools/decision_bench.py`: each `<task>/items.jsonl` holds the text a decision model reads
(`state`), its question (`noul`, `choice` or `score` with criteria), the ground truth (`gold`) and today's baseline
answer (`baseline_pred`). Running the benchmark needs only these files and a decision-model endpoint; no database.

| task | items | the decision | ground truth from |
|---|---|---|---|
| `routing` | 153 | which retrieval strategy a question needs | the question sets' own kinds |
| `answerability` | 178 | do these retrieved passages contain the answer | answer strings present, and unanswerable questions |
| `concept_a` | 240 | which of 8 vector-ranked Wikipedia categories fits an engineering article | the article's real categories |
| `concept_b` | 240 | which of 12 engineering fields an article belongs to | the field it was sampled from |
| `sysml_kind` | 101 | the SysML element kind of a model element's text | the Apollo 11 model |
| `alignment` | 113 | does this passage describe this model element | nanos3reader model provenance |
| `maturity` | 33 | concept / designed / implemented / verified | nanos3reader model maturity tags |

## Sources and licences of the text in the items

- English Wikipedia excerpts (routing, answerability, concept_a, concept_b): CC BY-SA 4.0, by Wikipedia
  contributors; each excerpt's article is named in its item.
- Apollo 11 SysML v2 model by Airbus (sysml_kind): MPL-2.0, https://github.com/airbus/apollo-11-sysml-v2.
- nanos3reader sources and its sysledge model (alignment, maturity, answerability): Apache-2.0, Yoav Bendor.
