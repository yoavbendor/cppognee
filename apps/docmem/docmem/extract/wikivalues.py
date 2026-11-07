"""wikiplain/2: wikiplain/1, with the templates that carry a value rendered as their text instead of dropped.

Before wikiplain/1's stripping, each template named in VALUE_TEMPLATES is replaced by its rendered wikitext, innermost
first, so links inside it keep their link records (the same marks as any wiki link). Nothing is converted: a
convert is its value and unit as the source writes them. Every other template is stripped as in wikiplain/1.

The allow-list and what each renders (examples in the right column):

  convert, cvt       value [connector value] unit           {{convert|65|%}} 65%; {{convert|10500|lbf|N}} 10500 lbf
                     (connector to, and, and(-), -, –, written as
                     the source has it; a second value after a
                     number makes a compound: 5 ft 6 in)
  val                value [e=exp] [u=unit]                 {{val|1.5|e=9|u=m}} 1.5 × 10^9 m
  nowrap, nobr       first argument                         {{nowrap|10 lbf}} 10 lbf
  frac, sfrac        a/b, or whole a b/c                    {{frac|1|2}} 1/2; {{frac|3|1|2}} 3 1/2
  circa              "c. " + first argument                 {{circa|1969}} c. 1969
  lang               second argument (the text)             {{lang|de|Mondlandung}} Mondlandung
  transl             second argument (the romanization)     {{transl|ja|Tsuki}} Tsuki
  endash, ndash, en dash  "–" (a range's dash inside a convert)  {{convert|10|{{endash}}|15|m}} 10–15 m
  abbr               first argument (the abbreviation)      {{abbr|NASA|...}} NASA
  mvar, math         the arguments joined with "|"          {{mvar|x}} x
  start date, end date, dts, birth date, death date
                     year[-month[-day]], months and days     {{start date|1969|7|16}} 1969-07-16
                     zero-padded; named arguments (df=...)
                     are ignored

A template whose arguments do not fit its pattern, or still hold an unrendered template, is left to the stripping, as
in wikiplain/1.
"""

import re

from . import wikiplain

EXTRACTOR = "wikiplain/2"
NUMBER = re.compile(r"[+\-~±]?\d[\d,.]*$")
RANGE_JOIN = {"to": " to ", "and": " and ", "and(-)": " and(-) ", "-": "-", "–": "–"}  # range connectors, as written
NO_SPACE_UNITS = ("%", "°")  # written against the number: 65%, 20°


def _name(t) -> str:
    return re.sub(r"[\s_]+", " ", str(t.name).strip()).lower()


def _args(t) -> tuple[list[str], dict[str, str]]:
    """(positional arguments, named arguments), each value as written and stripped."""
    pos = [str(p.value).strip() for p in t.params if not p.showkey]
    named = {str(p.name).strip(): str(p.value).strip() for p in t.params if p.showkey}
    return pos, named


def _with_unit(number: str, unit: str) -> str:
    if not unit:
        return number
    return number + unit if unit.startswith(NO_SPACE_UNITS) else f"{number} {unit}"


def _dash(pos, named):
    return "–"


def _convert(pos, named):
    if len(pos) < 2 or not pos[0] or not pos[1]:  # a value and its source unit are required
        return None
    if len(pos) >= 2 and pos[1] in RANGE_JOIN:  # value connector value unit
        if len(pos) < 4:
            return None
        return _with_unit(pos[0] + RANGE_JOIN[pos[1]] + pos[2], pos[3])
    if len(pos) >= 4 and NUMBER.match(pos[2]):  # compound: value unit value unit
        return f"{pos[0]} {pos[1]} {pos[2]} {pos[3]}"
    return _with_unit(pos[0], pos[1])


def _val(pos, named):
    if not pos or not pos[0]:
        return None
    text = pos[0]
    if named.get("e"):
        text += f" × 10^{named['e']}"
    if named.get("u"):
        text += f" {named['u']}"
    return text


def _first(pos, named):
    return pos[0] if pos and pos[0] else None


def _joined(pos, named):
    return "|".join(pos) if pos and pos[0] else None


def _frac(pos, named):
    if len(pos) == 2 and all(pos):
        return f"{pos[0]}/{pos[1]}"
    if len(pos) == 3 and all(pos):
        return f"{pos[0]} {pos[1]}/{pos[2]}"
    return None


def _circa(pos, named):
    return f"c. {pos[0]}" if pos and pos[0] else None


def _second(pos, named):
    return pos[1] if len(pos) >= 2 and pos[1] else None


def _abbr(pos, named):
    return _first(pos, named)


def _date(pos, named):
    parts = []
    for i, p in enumerate(pos[:3]):
        if not p:
            break
        parts.append(p.zfill(2) if i and p.isdigit() else p)
    return "-".join(parts) if parts else None


VALUE_TEMPLATES = {
    "convert": _convert,
    "cvt": _convert,
    "endash": _dash,  # a range's dash, as a convert connector
    "ndash": _dash,
    "en dash": _dash,
    "val": _val,
    "nowrap": _first,
    "nobr": _first,
    "frac": _frac,
    "sfrac": _frac,
    "circa": _circa,
    "lang": _second,
    "transl": _second,
    "abbr": _abbr,
    "mvar": _joined,
    "math": _joined,
    "start date": _date,
    "end date": _date,
    "dts": _date,
    "birth date": _date,
    "death date": _date,
}


def render_values(code) -> None:
    """Replace each allow-listed template of the parsed fragment code (nested ones too) by its rendered wikitext.
    Innermost first, so a parent sees its children already rendered. A template is left in place, to be stripped,
    when its arguments do not fit its pattern or still hold a template that is not rendered (a parser function
    such as {{#expr:...}} would otherwise leave its text behind)."""
    for t in reversed(list(code.filter_templates(recursive=True))):
        render = VALUE_TEMPLATES.get(_name(t))
        if render is None:
            continue
        text = render(*_args(t))
        if text is not None and "{{" not in text:
            code.replace(t, text)


def to_markdown_with_links(title: str, wikitext: str):
    """(markdown, links) as wikiplain.to_markdown_with_links, with the value templates rendered."""
    return wikiplain.markdown_with_links(title, wikitext, render_values)


def to_markdown(title: str, wikitext: str) -> str:
    return to_markdown_with_links(title, wikitext)[0]
