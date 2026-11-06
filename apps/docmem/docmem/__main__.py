import sys

USAGE = """usage: docmem <command> ...

  scan   cut markdown files into sections and passages (JSON lines on stdout)
  index  store a scan's stream (stdin) in cognee: vectors and provenance, no text
  ask    answer a question from the index, reading the passages back from the files
  eval   run a question set and score retrieval and answers
  stats  what the index holds and how big it is
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        return
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "scan":
        from .scan import main as m
    elif cmd == "index":
        from .index import main as m
    elif cmd == "ask":
        from .ask import main as m
    elif cmd == "eval":
        from .evaluate import main as m
    elif cmd == "stats":
        from .ask import stats_main as m
    else:
        raise SystemExit(USAGE)
    m(argv)


if __name__ == "__main__":
    main()
