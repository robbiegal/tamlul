"""Hebrew inverse text normalisation - a pure-Python port of the NeMo grammar.

Specification: NVIDIA NeMo-text-processing ``inverse_text_normalization/he``
(Apache-2.0, NVIDIA CORPORATION & AFFILIATES; licence text and provenance in
``data/nemo/LICENSE`` and ``data/nemo/VENDOR.md``): its TSV tables and test cases are vendored under ``data/nemo``
and ``tests/test_itn.py`` runs them. ``pynini`` (the FST runtime) has no
Windows wheel, hence the port. Known deviations from NeMo, all cosmetic:
punctuation stays attached to the converted number (NeMo's tokenizer emits
``20 , 22``), and adjacent numbers are separated by one space.

Behaviour summary (from the test suite):
* single digits (1-9), a bare ``עשר`` and a bare ``אפס`` stay as words unless
  accompanied by a minus, a unit, a magnitude word or a decimal point;
* everything >= 11 becomes digits with thousands separators (``1,000``),
  except years after ``שנת``/``בשנת``/``משנת``/``השנה`` (``1948``);
* a count before a trailing ``מיליון``/``מיליארד``/``אלף`` (with nothing after
  it) keeps the word (``10 מיליון``); ``מיליון`` alone or followed by more
  digits expands (``1,000,001``);
* decimals ``N נקודה d d d`` concatenate the fraction chunks (``1.453``);
  ``וחצי``/``ורבע`` after a single digit stay words (ambiguous with time)
  unless a unit/magnitude/minus follows;
* times need minutes or a day-part word; hours shift by ``בצהריים``/``בערב``/
  ``בלילה`` per the NeMo tables; ``חצות`` is 0;
* dates: ``day ב/ל month-name [year]`` or ``ordinal-day ל ordinal-month year``;
  ``month-name year``; two-word years (``עשרים עשרים ואחת`` -> 2021);
* measures: number + unit word -> digits + abbreviation (``ס״מ``, ``%``);
* whitelist phrases -> abbreviations (``ארצות הברית`` -> ``ארה״ב``);
* prefixes (ב, ל, כ, מ, ה, ו, ש, ...) glued to a converted token become
  ``prefix-`` (``כ-125``, ``ב-7:30``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources

# ----------------------------------------------------------------- tables


def _tsv(name: str) -> list[list[str]]:
    with resources.files("tamlul.itn").joinpath(f"data/nemo/{name}").open("r", encoding="utf-8") as fh:
        rows = []
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.strip():
                rows.append(line.split("\t"))
        return rows


DIGITS = {k: int(v) for k, v in _tsv("numbers_digit.tsv")}                 # 1-9, both genders
TEENS = {k: int(v) for k, v in _tsv("numbers_teen.tsv")}                    # 10-19 (multi-word keys)
TIES = {k: int(v) * 10 for k, v in _tsv("numbers_ties.tsv")}                # 20-90
THOUSANDS_CONSTRUCT = {k: int(v) for k, v in _tsv("numbers_thousands.tsv")}  # שלושת .. עשרת (x אלפים)
ZERO = {k for k, _ in _tsv("numbers_zero.tsv")}
HUNDREDS_EXC = {k: int(v) for k, v in _tsv("numbers_hundreds_exception.tsv")}   # מאה 1, מאתיים 2
THOUSANDS_EXC = {k: int(v) for k, v in _tsv("numbers_thousands_exception.tsv")}  # אלף 1, אלפיים 2
PREFIXES = sorted({r[0] for r in _tsv("prefix.tsv")}, key=len, reverse=True)
WHITELIST = [(r[1], r[0]) for r in _tsv("whitelist.tsv")]                   # phrase -> abbreviation
MEASURES = {r[1]: r[0] for r in _tsv("measurements.tsv")}                   # word -> symbol (attached)
SPACED_MEASURES = {r[1]: r[0] for r in _tsv("spaced_measurements.tsv")}     # word -> abbreviation (spaced)
MONTHS = {r[0]: int(r[1]) for r in _tsv("months_name2number.tsv")}
ORDINAL_MONTHS = {r[0]: int(r[1]) for r in _tsv("months_ordinal2number.tsv")}
ORDINAL_DIGIT = {r[0]: DIGITS.get(r[1], 10 if r[1] == "עשרה" else 0) for r in _tsv("ordinals_digit.tsv")}
FRACTIONS = {r[0]: r[1] for r in _tsv("numbers_decimal_fractions.tsv")}     # חצי -> "5", רבע -> "25", ...
FRACTION_EXC = {r[0] for r in _tsv("decimals_minutes_exception.tsv")}      # חצי, רבע (ambiguous with time)
HOUR_TO_NOON = {int(a): int(b) for a, b in _tsv("time_hour_to_noon.tsv")}
HOUR_TO_EVENING = {int(a): int(b) for a, b in _tsv("time_hour_to_evening.tsv")}
HOUR_TO_NIGHT = {int(a): int(b.strip()) for a, b in _tsv("time_hour_to_night.tsv")}
TO_HOUR = {r[0]: int(r[1]) for r in _tsv("time_to_hour.tsv")}              # "לשש" -> 5 (the hour before)
MINUTE_VERBOSE = {r[0]: int(r[1]) for r in _tsv("time_minute_verbose.tsv")}          # past the hour
MINUTE_TO_VERBOSE = {r[0]: int(r[1]) for r in _tsv("time_minute_to_verbose.tsv")}    # to the hour
VIABLE_HOURS = {r[0]: int(r[1]) for r in _tsv("numbers_viable_hours.tsv")}
DAY_SUFFIX = {r[0] for r in _tsv("time_day_suffix.tsv")}
EVENING_SUFFIX = {r[0] for r in _tsv("time_evening_suffix.tsv")}
NIGHT_SUFFIX = {r[0] for r in _tsv("time_night_suffix.tsv")}
NOON_SUFFIX = {r[0] for r in _tsv("time_noon_suffix.tsv")}
TIME_SUFFIXES = [s.split() for s in sorted(DAY_SUFFIX | EVENING_SUFFIX | NIGHT_SUFFIX | NOON_SUFFIX, key=len, reverse=True)]

MINUS = "מינוס"
POINT = "נקודה"
MILLION, BILLION, THOUSAND, THOUSANDS = "מיליון", "מיליארד", "אלף", "אלפים"
HUNDREDS_WORD = "מאות"
MINUTES_WORDS = {"דקות", "דקה"}
MIDNIGHT = "חצות"
YEAR_CUES = {"שנת", "בשנת", "משנת", "לשנת", "השנה", "בשנה", "שנה"}
HOUR_CUES = {"בשעה", "השעה", "שעה", "לשעה", "משעה"}
PUNCT = ".,;:!?\"'״׳()[]"
ORDINAL_UNITS = {**ORDINAL_DIGIT, "אחת עשרה": 11, "שתיים עשרה": 12}

_TOKEN_RE = re.compile(r"^([^\w֐-׿]*)(.*?)([^\w֐-׿]*)$", re.DOTALL)


@dataclass
class Tok:
    core: str            # the word without punctuation
    lead: str            # leading punctuation
    trail: str           # trailing punctuation
    raw: str


def _split_tok(t: str) -> Tok:
    m = _TOKEN_RE.match(t)
    lead, core, trail = (m.group(1), m.group(2), m.group(3)) if m else ("", t, "")
    return Tok(core=core, lead=lead, trail=trail, raw=t)


def _strip_vav(w: str) -> tuple[bool, str]:
    return (True, w[1:]) if len(w) > 1 and w.startswith("ו") else (False, w)


# ------------------------------------------------------------ number parse

def _unit_at(words: list[str], i: int, *, allow_vav: bool) -> tuple[int, int] | None:
    """Return (value, consumed) for a digit or teen at i (teens are one or two tokens)."""
    if i >= len(words):
        return None
    w = words[i]
    vav, w = _strip_vav(w) if allow_vav else (False, w)
    if vav and not allow_vav:
        return None
    if i + 1 < len(words):
        two = w + " " + words[i + 1]
        if two in TEENS:
            return TEENS[two], 2
    if w in TEENS:
        return TEENS[w], 1
    if w in DIGITS:
        return DIGITS[w], 1
    if w in ZERO:
        return 0, 1
    return None


def _small_at(words: list[str], i: int, *, allow_vav: bool) -> tuple[int, int, bool] | None:
    """Parse 0..99 at i. Returns (value, consumed, single_digit_flag)."""
    if i >= len(words):
        return None
    w = words[i]
    _, core = _strip_vav(w) if allow_vav else (False, w)
    if core in TIES:
        val, n = TIES[core], 1
        u = _unit_at(words, i + 1, allow_vav=True)
        if u and u[0] < 10 and words[i + 1].startswith("ו") and u[0] > 0:
            val += u[0]
            n += u[1]
        return val, n, False
    u = _unit_at(words, i, allow_vav=allow_vav)
    if u:
        bare = u[1] == 1 and (0 < u[0] < 10 or u[0] == 0 or (u[0] == 10 and words[i] in ("עשר", "עשרה")))
        return u[0], u[1], bare
    return None


def _hundreds_at(words: list[str], i: int, *, allow_vav: bool) -> tuple[int, int, bool] | None:
    """Parse 0..999 at i: [X מאות | מאה | מאתיים] [ו]small."""
    if i >= len(words):
        return None
    _, core = _strip_vav(words[i]) if allow_vav else (False, words[i])
    val, n = 0, 0
    if core in HUNDREDS_EXC:
        val, n = HUNDREDS_EXC[core] * 100, 1
    elif core in DIGITS and 3 <= DIGITS[core] <= 9 and i + 1 < len(words) and words[i + 1] == HUNDREDS_WORD:
        val, n = DIGITS[core] * 100, 2
    if n:
        s = _small_at(words, i + n, allow_vav=True)
        if s:
            val += s[0]
            n += s[1]
        return val, n, False
    s = _small_at(words, i, allow_vav=allow_vav)
    if s:
        return s
    return None


@dataclass
class Number:
    value: int
    consumed: int
    single_digit: bool           # 1-9 as one word, or bare עשר/אפס: stays a word unless forced
    trailing_magnitude: str = ""  # "מיליון"/"מיליארד"/"אלף" kept as a word after the count
    had_magnitude: bool = False


def parse_cardinal(words: list[str], i: int) -> Number | None:
    """Longest cardinal at i. Handles millions/billions/thousands with Hebrew construct forms."""
    n0 = i
    if i >= len(words):
        return None
    total = 0
    consumed = 0
    had_mag = False
    trailing = ""
    single = False
    # ---- billions / millions: [count] מיליארד/מיליון
    for mag_word, mag in ((BILLION, 10 ** 9), (MILLION, 10 ** 6)):
        j = i + consumed
        h = _hundreds_at(words, j, allow_vav=False)
        if h and j + h[1] < len(words) and words[j + h[1]] == mag_word:
            total += h[0] * mag
            consumed += h[1] + 1
            had_mag = True
            trailing = mag_word if h[0] else ""
            continue
        if j < len(words) and words[j] == mag_word and consumed == 0:
            total += mag
            consumed += 1
            had_mag = True
            trailing = ""
    # ---- thousands
    j = i + consumed
    allow_vav = consumed > 0
    vav, core = _strip_vav(words[j]) if (j < len(words) and allow_vav) else (False, words[j] if j < len(words) else "")
    if j < len(words):
        if core in THOUSANDS_EXC:
            total += THOUSANDS_EXC[core] * 1000
            consumed += 1
            had_mag = True
            trailing = THOUSAND if False else ""
        elif core in THOUSANDS_CONSTRUCT and j + 1 < len(words) and words[j + 1] == THOUSANDS:
            total += THOUSANDS_CONSTRUCT[core] * 1000
            consumed += 2
            had_mag = True
        else:
            h = _hundreds_at(words, j, allow_vav=allow_vav)
            if h and j + h[1] < len(words) and words[j + h[1]] == THOUSAND and h[0] >= 3:
                total += h[0] * 1000
                consumed += h[1] + 1
                had_mag = True
                trailing = THOUSAND if False else ""
    # ---- remainder 0..999
    j = i + consumed
    h = _hundreds_at(words, j, allow_vav=consumed > 0)
    if h:
        # a remainder after a magnitude must be joined by ו or be a hundreds/tie group; a bare
        # single digit after "מיליון"/"אלף" without ו is still accepted (מיליון שבעים -> 1,000,070)
        total += h[0]
        consumed += h[1]
        single = h[2] and not had_mag
        if had_mag:
            trailing = ""
    if consumed == 0:
        return None
    if had_mag and trailing and consumed == n0 - i + 0:  # unreachable guard
        pass
    return Number(value=total, consumed=consumed, single_digit=single, trailing_magnitude=trailing, had_magnitude=had_mag)


def _fmt(n: int, *, sep: bool = True) -> str:
    return f"{n:,}" if sep else str(n)


# ------------------------------------------------------------- decimals

def _parse_fraction_chunks(words: list[str], i: int) -> tuple[str, int] | None:
    """After נקודה: a sequence of 0..99 chunks; each chunk's digits are concatenated."""
    out = []
    n = 0
    while True:
        s = _small_at(words, i + n, allow_vav=False)
        if not s:
            break
        out.append(str(s[0]))
        n += s[1]
    if not out:
        return None
    return "".join(out), n


def _parse_verbal_fraction(words: list[str], i: int) -> tuple[str, int, bool] | None:
    """'וחצי', 'ורבע', 'ושלושת רבעי', 'ועשירית', 'ושתי עשיריות' -> (digits, consumed, ambiguous_with_time)."""
    if i >= len(words):
        return None
    vav, core = _strip_vav(words[i])
    if not vav:
        return None
    if i + 1 < len(words) and core + " " + words[i + 1] in FRACTIONS:
        return FRACTIONS[core + " " + words[i + 1]], 2, core in FRACTION_EXC
    if core in FRACTIONS:
        return FRACTIONS[core], 1, core in FRACTION_EXC
    return None


# ------------------------------------------------------------------ time

def _hour_at(words: list[str], i: int) -> tuple[int, int] | None:
    if i >= len(words):
        return None
    if words[i] == MIDNIGHT:
        return 0, 1
    if i + 1 < len(words) and words[i] + " " + words[i + 1] in VIABLE_HOURS:
        return VIABLE_HOURS[words[i] + " " + words[i + 1]], 2
    if words[i] in VIABLE_HOURS:
        return VIABLE_HOURS[words[i]], 1
    return None


def _to_hour_at(words: list[str], i: int) -> tuple[int, int] | None:
    """'לשש' / 'לאחת עשרה' / 'לחצות' -> the hour BEFORE, as in the NeMo table."""
    if i >= len(words) or not words[i].startswith("ל") or len(words[i]) < 2:
        return None
    core = words[i][1:]
    if i + 1 < len(words) and core + " " + words[i + 1] in TO_HOUR:
        return TO_HOUR[core + " " + words[i + 1]], 2
    if core in TO_HOUR:
        return TO_HOUR[core], 1
    return None


def _suffix_at(words: list[str], i: int) -> tuple[str, int] | None:
    for suf in TIME_SUFFIXES:
        if words[i:i + len(suf)] == suf:
            return " ".join(suf), len(suf)
    return None


def _shift(hour: int, suffix: str | None) -> int:
    if suffix is None:
        return hour
    if suffix in NOON_SUFFIX:
        return HOUR_TO_NOON.get(hour, hour)
    if suffix in EVENING_SUFFIX:
        return HOUR_TO_EVENING.get(hour, hour)
    if suffix in NIGHT_SUFFIX:
        return HOUR_TO_NIGHT.get(hour, hour)
    return hour


def _minutes_past_at(words: list[str], i: int) -> tuple[int, int] | None:
    """'וחצי' | 'ורבע' | 'ועשרה' | 'וחמישה' | 'ודקה' | 'ושתי דקות' | 'ו<N> דקות' -> (minutes, consumed)."""
    if i >= len(words):
        return None
    vav, core = _strip_vav(words[i])
    if not vav:
        return None
    if i + 1 < len(words) and core + " " + words[i + 1] in MINUTE_VERBOSE and words[i + 1] not in MINUTES_WORDS:
        m, n = MINUTE_VERBOSE[core + " " + words[i + 1]], 2
    elif core in MINUTE_VERBOSE:
        m, n = MINUTE_VERBOSE[core], 1
    else:
        s = _small_at([core] + words[i + 1:], 0, allow_vav=False)
        if not s:
            return None
        m, n = s[0], s[1]
        if not (i + n < len(words) and words[i + n] in MINUTES_WORDS):
            return None
    if i + n < len(words) and words[i + n] in MINUTES_WORDS:
        n += 1
    return m, n


def _minutes_to_at(words: list[str], i: int) -> tuple[int, int] | None:
    """'רבע' | 'עשרה' | 'חמישה' | 'עשרים' | 'עשרים וחמישה' | 'דקה' | '<N> דקות' followed by ל<hour>."""
    if i >= len(words):
        return None
    if i + 1 < len(words) and words[i] + " " + words[i + 1] in MINUTE_TO_VERBOSE:
        return MINUTE_TO_VERBOSE[words[i] + " " + words[i + 1]], 2
    if words[i] in MINUTE_TO_VERBOSE:
        return MINUTE_TO_VERBOSE[words[i]], 1
    s = _small_at(words, i, allow_vav=False)
    if s and i + s[1] < len(words) and words[i + s[1]] in MINUTES_WORDS:
        return 60 - s[0], s[1] + 1
    return None


def parse_time(words: list[str], i: int, *, cue_before: bool = False) -> tuple[str, int] | None:
    """Return ('H:MM', consumed) or None. The day-part suffix is peeked at, never consumed
    (NeMo keeps it: '7:30 בבוקר'). Needs explicit minutes, a suffix or חצות; 'בשעה חמש' stays words."""
    # minutes-to form: <minutes> ל<hour> [suffix]
    mt = _minutes_to_at(words, i)
    if mt:
        th = _to_hour_at(words, i + mt[1])
        if th:
            hour, n = th[0], mt[1] + th[1]
            suf = _suffix_at(words, i + n)
            hour = _shift(hour, suf[0] if suf else None)
            if words[i + mt[1]][1:] == MIDNIGHT:
                hour = 23
            return f"{hour}:{mt[0]:02d}", n
    h = _hour_at(words, i)
    if not h:
        return None
    hour, n = h
    mins = _minutes_past_at(words, i + n)
    minutes = 0
    explicit_minutes = False
    if mins:
        minutes = mins[0]
        n += mins[1]
        tail = words[i + h[1]]
        explicit_minutes = (i + h[1] + mins[1] - 1 < len(words) and words[i + h[1] + mins[1] - 1] in MINUTES_WORDS)             or tail[1:] in ("עשרה", "חמישה", "עשרים", "דקה")
    suf = _suffix_at(words, i + n)
    is_midnight = words[i] == MIDNIGHT
    if not (suf or is_midnight or explicit_minutes):
        return None   # 'שתיים וחצי' / 'שמונה ושלושת רבעי' alone: the decimal grammar decides
    hour = 0 if is_midnight else _shift(hour, suf[0] if suf else None)
    return f"{hour}:{minutes:02d}", n


# ------------------------------------------------------------------ dates

def _day_at(words: list[str], i: int) -> tuple[int, int] | None:
    """Cardinal (1-31) or ordinal day: ראשון, עשירי, אחת עשרה, עשרים ושישי ..."""
    if i >= len(words):
        return None
    if i + 1 < len(words) and words[i] + " " + words[i + 1] in ORDINAL_UNITS:
        return ORDINAL_UNITS[words[i] + " " + words[i + 1]], 2
    if words[i] in ORDINAL_UNITS:
        return ORDINAL_UNITS[words[i]], 1
    if words[i] in TIES and i + 1 < len(words):
        vav, core = _strip_vav(words[i + 1])
        if vav and core in ORDINAL_UNITS and ORDINAL_UNITS[core] < 10:
            return TIES[words[i]] + ORDINAL_UNITS[core], 2
    s = _small_at(words, i, allow_vav=False)
    if s and 1 <= s[0] <= 31:
        return s[0], s[1]
    return None


def _month_at(words: list[str], i: int) -> tuple[int, int, bool] | None:
    """'במאי' / 'לאפריל' (name) or 'לשלישי' / 'לשתיים עשרה' (ordinal). Returns (month, consumed, is_name)."""
    if i >= len(words):
        return None
    w = words[i]
    for pre in ("ב", "ל"):
        if w.startswith(pre) and len(w) > 1:
            core = w[1:]
            if core in MONTHS:
                return MONTHS[core], 1, True
            if pre == "ל":
                if i + 1 < len(words) and core + " " + words[i + 1] in ORDINAL_MONTHS:
                    return ORDINAL_MONTHS[core + " " + words[i + 1]], 2, False
                if core in ORDINAL_MONTHS:
                    return ORDINAL_MONTHS[core], 1, False
    return None


def _year_at(words: list[str], i: int, *, allow_two_digit: bool) -> tuple[str, int] | None:
    """Full cardinal year (1948, 2015), 'עשרים עשרים ואחת' (2021), or two digits (24)."""
    if i >= len(words):
        return None
    if words[i] in TIES and TIES[words[i]] == 20:
        s = _small_at(words, i + 1, allow_vav=False)
        if s and (s[0] >= 10 or s[1] >= 1) and not (i + 1 + s[1] < len(words) and words[i + 1 + s[1]] == THOUSAND):
            if s[0] < 100:
                return f"20{s[0]:02d}", 1 + s[1]
    n = parse_cardinal(words, i)
    if n and n.value >= 1000 and not n.trailing_magnitude:
        return str(n.value), n.consumed
    if allow_two_digit and n and 10 <= n.value < 100:
        return f"{n.value:02d}", n.consumed
    return None


def parse_date(words: list[str], i: int) -> tuple[str, int] | None:
    d = _day_at(words, i)
    if d:
        m = _month_at(words, i + d[1])
        if m:
            n = d[1] + m[1]
            y = _year_at(words, i + n, allow_two_digit=True)
            if y:
                return f"{d[0]}.{m[0]}.{y[0]}", n + y[1]
            if m[2]:  # month by name: year optional
                return f"{d[0]}.{m[0]}", n
            return None
    # month-name + year
    w = words[i]
    core = w[1:] if w[:1] in ("ב", "מ", "ל") and w[1:] in MONTHS else (w if w in MONTHS else None)
    if core:
        y = _year_at(words, i + 1, allow_two_digit=False)
        if y:
            return f"{w} {y[0]}", 1 + y[1]
    return None


# ---------------------------------------------------------------- driver

@dataclass
class Span:
    start: int
    end: int
    text: str
    prefix: str = ""


def _prefix_split(word: str) -> list[tuple[str, str]]:
    """All (prefix, rest) splits to try, longest prefix first, plus ('', word)."""
    out = []
    for p in PREFIXES:
        if word.startswith(p) and len(word) > len(p) + 1:
            out.append((p, word[len(p):]))
    out.append(("", word))
    return out


def _measure_at(words: list[str], i: int) -> tuple[str, int, bool] | None:
    """Unit word(s) at i -> (symbol, consumed, spaced)."""
    for table, spaced in ((SPACED_MEASURES, True), (MEASURES, False)):
        for k in sorted(table, key=lambda s: -len(s)):
            parts = k.split()
            if words[i:i + len(parts)] == parts:
                return table[k], len(parts), spaced
    return None


def _try_number_like(words: list[str], i: int, cues_before: str) -> Span | None:
    """Time, date, then decimal/measure/cardinal at i (i already has its prefix removed)."""
    if cues_before in YEAR_CUES:
        y = _year_at(words, i, allow_two_digit=False)
        if y:
            return Span(i, i + y[1], y[0])
    t = parse_time(words, i, cue_before=cues_before in HOUR_CUES)
    if t:
        return Span(i, i + t[1], t[0])
    d = parse_date(words, i)
    if d:
        return Span(i, i + d[1], d[0])
    minus = words[i] == MINUS
    j = i + 1 if minus else i
    n = parse_cardinal(words, j)
    if not n:
        return None
    end = j + n.consumed
    forced = minus
    text = ""
    # decimals
    if end < len(words) and words[end] == POINT:
        frac = _parse_fraction_chunks(words, end + 1)
        if frac:
            text = f"{_fmt(n.value)}.{frac[0]}"
            end += 1 + frac[1]
            forced = True
    if not text:
        vf = _parse_verbal_fraction(words, end)
        if vf:
            digits, k, ambiguous = vf
            unit_after = end + k < len(words) and (_measure_at(words, end + k) or words[end + k] in (MILLION, BILLION, THOUSAND))
            if ambiguous and n.single_digit and not minus and not unit_after:
                pass  # "שתיים וחצי" stays (time/decimal ambiguity)
            else:
                text = f"{_fmt(n.value)}.{digits}"
                end += k
                forced = True
    magnitude = ""
    if text and end < len(words) and words[end] in (MILLION, BILLION, THOUSAND):
        magnitude = words[end]
        end += 1
    if not text:
        text = _fmt(n.value, sep=cues_before not in YEAR_CUES)
        if n.trailing_magnitude:
            text = f"{_fmt(n.value // (10**9 if n.trailing_magnitude == BILLION else 10**6))} {n.trailing_magnitude}"
            forced = True
    # units
    m = _measure_at(words, end)
    if m:
        symbol, k, spaced = m
        text = f"{text} {symbol}" if spaced else f"{text}{symbol}"
        end += k
        forced = True
    if magnitude:
        text = f"{text} {magnitude}"
    if n.single_digit and not forced and not n.had_magnitude:
        return None
    if minus:
        text = "-" + text
    return Span(i, end, text)


def _unit_first_at(words: list[str], i: int) -> Span | None:
    """'אחוז אחד' / 'סנטימטר אחד' -> '1%' / '1 ס״מ'."""
    m = _measure_at(words, i)
    if not m:
        return None
    symbol, k, spaced = m
    if i + k < len(words) and words[i + k] in DIGITS and DIGITS[words[i + k]] == 1:
        return Span(i, i + k + 1, f"1 {symbol}" if spaced else f"1{symbol}")
    return None


def _whitelist_at(words: list[str], i: int) -> Span | None:
    best = None
    for phrase, abbr in WHITELIST:
        parts = phrase.split()
        if words[i:i + len(parts)] == parts and (best is None or len(parts) > best.end - best.start):
            best = Span(i, i + len(parts), abbr)
    return best


def itn_tokens(tokens: list[str]) -> list[str]:
    """Token-preserving ITN: converted spans collapse into the first token, the rest become ''."""
    toks = [_split_tok(t) for t in tokens]
    cores = [t.core for t in toks]
    out = list(tokens)
    i = 0
    n = len(cores)
    while i < n:
        span = None
        if cores[i]:
            span = _whitelist_at(cores, i) or _unit_first_at(cores, i)
            if span is None:
                prev = cores[i - 1] if i > 0 else ""
                for prefix, rest in _prefix_split(cores[i]):
                    trial = cores[:i] + [rest] + cores[i + 1:]
                    s = _try_number_like(trial, i, prev)
                    if s:
                        if prefix and s.text and s.text[0].isdigit() or (prefix and s.text.startswith("-")):
                            s.prefix = prefix
                        elif prefix:
                            continue  # prefix + non-numeric result (e.g. month word): skip
                        span = s
                        break
        if span is None:
            i += 1
            continue
        first, last = toks[span.start], toks[span.end - 1]
        text = span.text
        if span.prefix:
            text = f"{span.prefix}-{text}"
        out[span.start] = f"{first.lead}{text}{last.trail}"
        for k in range(span.start + 1, span.end):
            out[k] = ""
        i = span.end
    return out


def itn_text(text: str) -> str:
    toks = text.split()
    return " ".join(t for t in itn_tokens(toks) if t != "")
