"""
User-editable word corrections for transcripts.

Speech models (Voxtral in particular, which has no hotword support) reliably
mis-hear proper names — brands, products, programme names. This applies a
simple, predictable find/replace list the user maintains, e.g. turning
"Flor-Öle" into "VitaFlor" or "Sonvida" into "Sonvita".

The list lives in the noScribe config directory as `voxtral_corrections.yml`.
Format (case-insensitive, whole-word matches):

    - to: VitaFlor
      from: [vitaflor, "vita flor", "flor-öl", "flor-öle"]
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

CORRECTIONS_FILENAME = "voxtral_corrections.yml"

DEFAULT_CORRECTIONS = """\
# noScribe - word corrections for the Voxtral engine
#
# Every "from" variant is replaced by the "to" value (case-insensitive,
# whole words/phrases only). Useful for brand, product and programme names
# the model keeps mis-hearing. One entry may list several spellings.
#
# The file starts empty on purpose - corrections are personal to your
# material. Uncomment and adapt the examples to get started:
#
# - to: VitaFlor
#   from: [vitaflor, "vita flor", "flor-öl", "flor-öle"]
# - to: Sonvita
#   from: [sonvida, sonvieda, sonwita, "son vita"]
"""


def ensure_default_file(config_dir):
    """Create the corrections file with a documented default if it is missing.
    Returns the path to the file."""
    path = os.path.join(config_dir, CORRECTIONS_FILENAME)
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CORRECTIONS)
        except Exception as e:
            logger.warning("Could not create corrections file %s: %s", path, e)
    return path


def load_corrections(path):
    """Load the corrections file into a list of (compiled_regex, replacement)."""
    if not path or not os.path.exists(path):
        return []
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
    except Exception as e:
        logger.warning("Could not read corrections file %s: %s", path, e)
        return []
    if not isinstance(data, list):
        logger.warning("Corrections file %s must contain a list of entries.", path)
        return []

    rules = []
    for entry in data:
        # Tell the user about a mistyped entry instead of silently skipping it —
        # this file is hand-maintained and typos would otherwise look like the
        # feature not working.
        if not isinstance(entry, dict):
            logger.warning("Ignoring invalid corrections entry (not a mapping): %r", entry)
            continue
        to = entry.get("to")
        frm = entry.get("from") or []
        if isinstance(frm, str):
            frm = [frm]
        frm = [f for f in frm if f]
        if not to or not frm:
            logger.warning("Ignoring incomplete corrections entry (needs 'to' and 'from'): %r", entry)
            continue
        # Longest patterns first so "vita flor oil" wins over "vita flor".
        pats = sorted((re.escape(str(f)) for f in frm), key=len, reverse=True)
        rx = re.compile(r"(?<!\w)(?:" + "|".join(pats) + r")(?!\w)", re.IGNORECASE)
        rules.append((rx, str(to)))
    return rules


def _koelner_phonetik(word):
    """Cologne phonetics code of a word (German equivalent of Soundex).

    Words that are pronounced alike get the same code, which is exactly the
    class of error a speech model makes on names: it hears the sound correctly
    but cannot know the spelling ("Marcus"/"Markus") or picks a similar-sounding
    one ("Muna"/"Mona").
    """
    w = word.lower()
    w = (w.replace("ä", "a").replace("ö", "o").replace("ü", "u")
          .replace("ß", "ss").replace("é", "e").replace("è", "e").replace("ê", "e"))
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return ""
    codes = []
    for i, ch in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        prv = w[i - 1] if i else ""
        if ch in "aeiouy":
            c = "0"
        elif ch == "b" or (ch == "p" and nxt != "h"):
            c = "1"
        elif ch in "dt" and nxt not in "csz":
            c = "2"
        elif ch in "fvw" or (ch == "p" and nxt == "h"):
            c = "3"
        elif ch in "gkq":
            c = "4"
        elif ch == "c":
            # "c" is spoken like "k" before a/h/k/l/o/q/r/u/x (and at the start),
            # otherwise like "z" -- this is what maps Marcus onto Markus.
            c = "4" if (nxt in "ahkloqrux" and (i == 0 or prv not in "sz")) else "8"
        elif ch == "x":
            c = "48"
        elif ch == "l":
            c = "5"
        elif ch in "mn":
            c = "6"
        elif ch == "r":
            c = "7"
        elif ch in "sz":
            c = "8"
        else:
            continue
        codes.append(c)
    out = "".join(codes)
    if not out:
        # Every letter was uncoded (h/j only, e.g. "Jhh") -> no phonetic code.
        # Return "" rather than indexing out[0] into an empty string (IndexError
        # that would otherwise abort the whole correction pass).
        return ""
    out = re.sub(r"(.)\1+", r"\1", out)          # collapse repeats
    return out[0] + out[1:].replace("0", "")      # drop vowels except a leading one


def _edit_distance(a, b, limit=2):
    """Levenshtein distance, capped: returns limit+1 once it is exceeded."""
    a, b = a.lower(), b.lower()
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def apply_name_corrections(text, names, language=None):
    """Normalise spoken names to the spelling the user provided.

    A speech model has no way to know whether it should write "Markus" or
    "Marcus", and it easily picks a similar-sounding spelling ("Muna" for
    "Mona"). Where the user has already told us the names (the speaker-names
    field), both can be fixed.

    Cologne phonetics models *German* pronunciation, so this only runs on German
    audio. It is deliberately not generalised: a language-neutral "close
    spelling" rule was tried and rewrote ordinary words ("Rome" -> "Mona",
    "Anna" -> "Anne"), so for any other language -- and for auto-detect, where we
    do not know what was spoken -- nothing is changed here and the user's own
    correction list remains the way to fix names.

    A replacement additionally requires the same word length, which is what
    keeps ordinary words intact -- the phonetic code collapses vowels, so
    "Rom"/"Ruhm" would otherwise be rewritten to "Mona".

    Residual risk to be aware of: a *different* name that sounds the same as a
    speaker's ("Anna" next to a speaker called "Anne") is rewritten too. Only
    names the user explicitly entered are ever used as targets.
    """
    if not str(language or "").strip().lower().startswith("de"):
        return text
    names = [n for n in (names or []) if len(n) >= 3 and n.isalpha()]
    if not names:
        return text
    wanted = {}
    for n in names:
        code = _koelner_phonetik(n)
        if code:
            wanted.setdefault(code, n)

    def repl(m):
        word = m.group(0)
        name = wanted.get(_koelner_phonetik(word))
        if not name or word == name:
            return word
        if len(word) != len(name) or _edit_distance(word, name) > 2:
            return word
        return name

    # Only capitalised words: names appear capitalised, and this keeps the rule
    # away from lowercase everyday words that happen to sound similar.
    return re.sub(r"(?<!\w)[A-ZÄÖÜ][\wäöüß]{2,}", repl, text)


def apply_corrections(text, rules):
    """Apply the compiled correction rules to `text`."""
    for rx, to in rules:
        # A callable replacement treats `to` as literal text — a backslash or
        # "\1" in a user-supplied value must never be a regex template.
        text = rx.sub(lambda m, to=to: to, text)
    return text
