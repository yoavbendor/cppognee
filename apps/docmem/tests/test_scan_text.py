"""Scan of code and text files (docmem scan --ext / --name / --include-hidden): no headings, blocks cut at blank
lines (long ones at line ends), path [file name], no links, and the file selection. No cognee, no network.

Run from apps/docmem:  python -m pytest tests/test_scan_text.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmem.scan import blocks, is_text, iter_files, passages_of, scan_bytes, scan_file, split_lines  # noqa: E402


def _kinds(data: bytes, headings: bool):
    return [(kind, lvl, t) for kind, lvl, t, *_ in blocks(data, headings=headings)]


def test_hash_comments_are_headings_only_for_markdown():
    cmake = b"# ---------\n# Build options\ncmake_minimum_required(VERSION 3.16)\n"
    assert _kinds(cmake, headings=True)[0] == ("heading", 1, "---------")
    assert all(kind == "block" for kind, _, _ in _kinds(cmake, headings=False))


def test_fences_are_not_recognised_in_code():
    data = b"```\nint x;\n\nint y;\n```\n"
    assert all(kind == "block" for kind, _, _ in _kinds(data, headings=False))
    assert _kinds(data, headings=True)[0][0] == "block"  # the fence opens a markdown block


def test_text_file_is_one_section_path_is_its_name_and_it_has_no_links(tmp_path):
    f = tmp_path / "s3_reader.cpp"
    f.write_text("// header comment\nint f() {\n  return 1;\n}\n\nint g() {\n  return 2;\n}\n")
    recs = list(scan_file(f, 180, 320, text=True))
    assert [r["type"] for r in recs].count("section") == 0
    passages = [r for r in recs if r["type"] == "passage"]
    assert passages and all(r["path"] == ["s3_reader.cpp"] for r in passages)
    assert not any(r["type"] == "link" for r in recs)
    assert recs[0]["title"] == "s3_reader.cpp"
    # byte ranges are exact: each passage's bytes are its text
    data = f.read_bytes()
    for r in passages:
        assert data[r["offset"]:r["offset"] + r["length"]].decode() == r["text"]
        assert r["line"] >= 1 and r["end_line"] >= r["line"]


def test_long_code_block_is_cut_at_line_ends_not_inside_a_line():
    data = b"\n".join(b"int v%d = %d;" % (i, i) for i in range(50)) + b"\n"
    pieces = split_lines(data, 0, len(data), max_words=20, target_words=10)
    assert len(pieces) > 1
    for a, b in pieces:
        text = data[a:b].decode()
        assert text.startswith("int v") and text.rstrip().endswith(";")
    assert [t for a, b in pieces for t in data[a:b].split()] == data.split()  # no line lost or split


def test_markdown_path_still_uses_sentence_cuts_and_headings():
    data = b"# Title\n\nIntro text here.\n\n## Part\n\nBody text.\n"
    recs = list(scan_bytes(data, {"uri": "file:///x/a.md", "sha256": "", "size": 0, "mtime": 0, "title": None},
                           "a", 180, 320, links=[]))
    assert [r["path"] for r in recs if r["type"] == "section"] == [["Title"], ["Title", "Part"]]


def test_passages_of_default_split_is_the_sentence_split():
    data = b"One sentence. Another sentence follows here. And a third one closes it all."
    items = [(0, len(data), 1)]
    pieces = passages_of(data, items, target_words=5, max_words=5)  # split_long, the markdown default
    assert len(pieces) >= 2


def test_iter_files_selects_markdown_and_named_code_files_and_skips_hidden_unless_asked(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.cpp").write_text("int x;\n")
    (tmp_path / "src" / "x.h").write_text("int h;\n")
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    (tmp_path / "notes.bin").write_text("no\n")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    names = lambda files: sorted(f.relative_to(tmp_path).as_posix() for f in files)  # noqa: E731

    default = names(iter_files([tmp_path], exts=(".cpp",), names=("CMakeLists.txt",)))
    assert default == ["CMakeLists.txt", "docs/a.md", "src/x.cpp"]
    hidden = names(iter_files([tmp_path], exts=(".yml",), names=(), include_hidden=True))
    assert ".github/workflows/ci.yml" in hidden
    assert ".github/workflows/ci.yml" not in names(iter_files([tmp_path], exts=(".yml",)))


def test_is_text_follows_the_suffix_and_name_lists():
    assert is_text(Path("a/x.CPP"), exts=(".cpp",))
    assert is_text(Path("proj/CMakeLists.txt"), names=("CMakeLists.txt",))
    assert not is_text(Path("proj/readme.md"), exts=(".cpp",), names=("CMakeLists.txt",))
