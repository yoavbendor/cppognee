import sys

USAGE = """usage: docmem <command> ...

  scan   cut markdown files into sections and passages (JSON lines on stdout)
  scan-wiki  a random sample of English Wikipedia articles, read from the dump (JSON lines on stdout)
  scan-sysml a SysML v2 textual model as JSON lines, for a model DB only (never a prose DB)
  model-resolve  resolve a model DB's references to qualified names; write the edges and metaclasses (no LLM)
  model-show     one model element: metaclass, owner, types, owned and inherited features
  model-export   a model DB's resolved model as a nomograph index (for sysmldiag and nomograph-sysml)
  model-check    nomograph-sysml's completeness checks on a model, stored as findings in the model DB
  model-gaps     the stored model findings, by check and under a qualified name
  index  store a scan's stream (stdin) in cognee: vectors and provenance, no text
  ask    answer a question from the index, reading the passages back from the files
  eval   run a question set and score retrieval and answers
  stats  what the index holds and how big it is
  scan-wiki  random Wikipedia articles from the dump (by HTTP Range), linked to their revisions
  refresh    close the index's gaps with one more read per source
  improve    link entities across documents, group topics (stored records only, no source reads)
  kwsearch   BM25 keyword search over an index's passages (no vectors, no LLM)
  ontology   load a SKOS concept scheme (IPTC Media Topics JSON or SKOS RDF) into an index (no LLM)
  classify   link documents and sections to their nearest concepts by stored vectors (no LLM)
  concepts   a document's concepts with scores, or the top levels of the concept tree
  wiki-categories  the Wikipedia category tree of the indexed articles as a SKOS file (dump read, no LLM)
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        return
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "scan":
        from .scan import main as m
    elif cmd == "scan-wiki":
        from .scan_wiki import main as m
    elif cmd == "scan-sysml":
        from .scan_sysml import main as m
    elif cmd == "model-resolve":
        from .sysml_resolve import resolve_main as m
    elif cmd == "model-show":
        from .sysml_resolve import show_main as m
    elif cmd == "model-export":
        from .model_export import export_main as m
    elif cmd == "model-check":
        from .model_check import check_main as m
    elif cmd == "model-gaps":
        from .model_check import gaps_main as m
    elif cmd == "index":
        from .index import main as m
    elif cmd == "ask":
        from .ask import main as m
    elif cmd == "eval":
        from .evaluate import main as m
    elif cmd == "refresh":
        from .refresh import main as m
    elif cmd == "improve":
        from .improve import main as m
    elif cmd == "stats":
        from .ask import stats_main as m
    elif cmd == "kwsearch":
        from .keywords import main as m
    elif cmd == "ontology":
        from .ontology import ontology_main as m
    elif cmd == "classify":
        from .ontology import classify_main as m
    elif cmd == "concepts":
        from .ontology import concepts_main as m
    elif cmd == "wiki-categories":
        from .wiki_categories import main as m
    else:
        raise SystemExit(USAGE)
    m(argv)


if __name__ == "__main__":
    main()
