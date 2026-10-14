"""Benchmark datasets: pinned sources, checksums, BIO parsing and label verbalization.

Every file is downloaded from its original distribution (pinned to a commit
where the host supports it) and verified against a SHA-256 checksum, so a run
either uses exactly the bytes these results were produced with or fails.

Sources
-------
- CoNLL-2003 and CrossNER (AI, literature, music, politics, science):
  https://github.com/zliucr/CrossNER (Liu et al., AAAI 2021), ``ner_data/``.
  The CoNLL-2003 copy in that repository uses the CrossNER tag names
  (``person``, ``organisation``, ``location``, ``misc``).
- MIT Movie (``engtest.bio``, 12 types) and MIT Restaurant (``restauranttest.bio``):
  https://groups.csail.mit.edu/sls/downloads/ (Liu et al., ICASSP 2013).
  Not versioned by the host; the checksum is the only pin.
- WNUT-17: https://github.com/leondz/emerging_entities_17 (Derczynski et al., 2017).

Label verbalization
-------------------
GLiNER is prompted with natural-language entity types. The mapping from raw
tags to prompts below is ours. The GLiNER paper evaluates on the
InstructUIE/UniNER JSON release of the same test sets, whose exact label
strings may differ, so absolute F1 here is not directly comparable to the
paper's table.
"""

from __future__ import annotations

import hashlib
import urllib.request
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import field, dataclass

CROSSNER_REV = "2e7ba2a7798c961e3f29fbc51252c5a8d40224bf"
WNUT_REV = "e52a2d2a71ac1a051ca2d532eef28653c5c02603"
_CROSSNER = f"https://raw.githubusercontent.com/zliucr/CrossNER/{CROSSNER_REV}/ner_data"
_WNUT = f"https://raw.githubusercontent.com/leondz/emerging_entities_17/{WNUT_REV}"
_MIT = "https://groups.csail.mit.edu/sls/downloads"

DEFAULT_CACHE = Path("data/raw")

Example = Dict[str, object]  # {"tokenized_text": List[str], "ner": List[Tuple[int, int, str]]}

_CROSSNER_LABELS = {
    "academicjournal": "academic journal",
    "astronomicalobject": "astronomical object",
    "chemicalcompound": "chemical compound",
    "chemicalelement": "chemical element",
    "literarygenre": "literary genre",
    "musicalartist": "musical artist",
    "musicalinstrument": "musical instrument",
    "musicgenre": "music genre",
    "politicalparty": "political party",
    "programlang": "programming language",
    "organisation": "organization",
    "misc": "miscellaneous",
}

_MIT_MOVIE_LABELS = {
    "ACTOR": "actor",
    "CHARACTER": "character",
    "DIRECTOR": "director",
    "GENRE": "genre",
    "PLOT": "plot",
    "RATING": "mpaa rating",
    "RATINGS_AVERAGE": "average ratings",
    "REVIEW": "review",
    "SONG": "song",
    "TITLE": "title",
    "TRAILER": "trailer",
    "YEAR": "year",
}

_MIT_RESTAURANT_LABELS = {
    "Amenity": "amenity",
    "Cuisine": "cuisine",
    "Dish": "dish",
    "Hours": "hours",
    "Location": "location",
    "Price": "price",
    "Rating": "rating",
    "Restaurant_Name": "restaurant name",
}

_WNUT_LABELS = {"creative-work": "creative work"}


@dataclass(frozen=True)
class Split:
    """One downloadable file of a dataset."""

    url: str
    sha256: str


@dataclass(frozen=True)
class DatasetSpec:
    """A benchmark dataset: where to get it and how to read it."""

    name: str
    splits: Dict[str, Split]
    tag_first: bool  # MIT files are "TAG<TAB>token", the others "token<TAB>TAG"
    label_map: Dict[str, str] = field(default_factory=dict)
    zero_shot: bool = True

    def verbalize(self, raw: str) -> str:
        return self.label_map.get(raw, raw.lower())


def _crossner(domain: str, hashes: Dict[str, str], zero_shot: bool = True) -> DatasetSpec:
    name = "conll2003" if domain == "conll2003" else f"crossner_{domain}"
    return DatasetSpec(
        name=name,
        splits={sp: Split(f"{_CROSSNER}/{domain}/{sp}.txt", h) for sp, h in hashes.items()},
        tag_first=False,
        label_map=_CROSSNER_LABELS,
        zero_shot=zero_shot,
    )


DATASETS: Dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        _crossner(
            "conll2003",
            {
                "train": "098791b165293cac55aa60ea5395b48ed441bc962c00278aa4789dce68624bf8",
                "dev": "c8869cec4f419e7b635843aae285990581bf4685b7193c81696d87e1939f0c63",
                "test": "18d9d738defc506ebc5d6748cb580245327ef5c4357b026d7d9ba7c6e55fe841",
            },
            zero_shot=False,
        ),
        _crossner(
            "ai",
            {
                "dev": "b47ffda6de5901a454a873feac1bebe782e54ce7edf9deb729cab3d769212eee",
                "test": "7595ad386e0a98067100fe47f86811b38867a236011fee97cfaa90f778c37bcf",
            },
        ),
        _crossner(
            "literature",
            {
                "dev": "99281a51943cf7e3c2a58b3dc45c0e5ab6d329624ebf2ba6b4801c428396162f",
                "test": "77267f8b3e8afe993865c64cfc605f65671f6a0ac9cba65de970f12180fed20e",
            },
        ),
        _crossner(
            "music",
            {
                "dev": "e256f1949df52a61a3bb60095a0f41c1060843f1014b0f3d22d8371611b72e78",
                "test": "c66e345f959ac65593f76c6b1bd8c6d75f2a6f0bc197389b2e6700d101fd094c",
            },
        ),
        _crossner(
            "politics",
            {
                "dev": "4f1f30eb832f4c5fe474f67ca45e875d863bddf0bbe432262d272ef13639fa8f",
                "test": "05cadc5e129d394f7f92b4a052cf4b6ea4c60878d32a833e500cf046c297455f",
            },
        ),
        _crossner(
            "science",
            {
                "dev": "87ec141c7b41b3b52ec861737c4030d65c6e53954ed3d8106baca8603bbf8f24",
                "test": "3616d59d4d2b9dbf4c1126f8d8ba47ff97886223c95f93649bfd527343e64ced",
            },
        ),
        DatasetSpec(
            name="mit_movie",
            splits={
                "test": Split(
                    f"{_MIT}/movie/engtest.bio", "2ba7c6da2e5c1d897ebbcbcefad9e0f03eed41b183106e7f26b7f24b704282fd"
                )
            },
            tag_first=True,
            label_map=_MIT_MOVIE_LABELS,
        ),
        DatasetSpec(
            name="mit_restaurant",
            splits={
                "test": Split(
                    f"{_MIT}/restaurant/restauranttest.bio",
                    "09809dd126a3781c4205d06e721752054e51a10b1c98f3495ef17cd723f25ce2",
                )
            },
            tag_first=True,
            label_map=_MIT_RESTAURANT_LABELS,
        ),
        DatasetSpec(
            name="wnut17",
            splits={
                "train": Split(
                    f"{_WNUT}/wnut17train.conll", "731820e13f71af324c6b55a1575ec2ce59fbaa2a0806f8f0400b98d56cd6a7a5"
                ),
                "dev": Split(
                    f"{_WNUT}/emerging.dev.conll", "e053b752b8155113ca66e7db08af5f2f5f43ab993694e4774f1ec5bad42312d8"
                ),
                "test": Split(
                    f"{_WNUT}/emerging.test.annotated",
                    "2aa79b764e56ec9264a1b30fdd9b70195bd00ff400b62edd8f399d5f13c178f0",
                ),
            },
            tag_first=False,
            label_map=_WNUT_LABELS,
        ),
    ]
}

ZERO_SHOT_BENCHMARKS: List[str] = [
    "wnut17",
    "crossner_ai",
    "crossner_literature",
    "crossner_music",
    "crossner_politics",
    "crossner_science",
    "mit_movie",
    "mit_restaurant",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, split: str, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Download one split (if needed) and verify its checksum."""
    spec = DATASETS[name].splits[split]
    path = Path(cache_dir) / f"{name}_{split}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(spec.url, tmp)
        tmp.rename(path)
    digest = _sha256(path)
    if digest != spec.sha256:
        raise RuntimeError(f"Checksum mismatch for {path}: expected {spec.sha256}, got {digest}. Delete it and retry.")
    return path


def parse_bio(text: str, tag_first: bool) -> List[Tuple[List[str], List[str]]]:
    """Parse a two-column BIO file into ``(tokens, tags)`` sentences.

    Sentences are separated by blank lines. Lines that do not have exactly two
    tab-separated fields are rejected rather than silently skipped.
    """
    sentences: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    tags: List[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r\n")
        if not line.strip():
            if tokens:
                sentences.append((tokens, tags))
                tokens, tags = [], []
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"line {lineno}: expected 2 tab-separated fields, got {len(parts)}: {line!r}")
        tag, tok = (parts[0], parts[1]) if tag_first else (parts[1], parts[0])
        tokens.append(tok)
        tags.append(tag.strip())
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def bio_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """Convert BIO tags to inclusive ``(start, end, type)`` word spans.

    An ``I-X`` that does not continue an ``X`` entity starts a new entity
    (the conlleval convention).
    """
    spans: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    current: Optional[str] = None
    for i, tag in enumerate(tags):
        prefix, _, typ = tag.partition("-")
        if tag == "O" or prefix not in ("B", "I"):
            if current is not None:
                spans.append((start, i - 1, current))
            start, current = None, None
            continue
        if prefix == "B" or typ != current:
            if current is not None:
                spans.append((start, i - 1, current))
            start, current = i, typ
    if current is not None:
        spans.append((start, len(tags) - 1, current))
    return spans


def load(name: str, split: str, cache_dir: Path = DEFAULT_CACHE) -> Tuple[List[Example], List[str]]:
    """Load a split in GLiNER format.

    Returns:
        ``(examples, labels)`` where each example is
        ``{"tokenized_text": [...], "ner": [(start, end, label), ...]}`` and
        ``labels`` is the sorted list of verbalized entity types of the split.
    """
    spec = DATASETS[name]
    text = fetch(name, split, cache_dir).read_text(encoding="utf-8")
    examples: List[Example] = []
    label_set = set()
    for tokens, tags in parse_bio(text, spec.tag_first):
        ner = [(s, e, spec.verbalize(t)) for s, e, t in bio_to_spans(tags)]
        label_set.update(lab for _, _, lab in ner)
        examples.append({"tokenized_text": tokens, "ner": ner})
    return examples, sorted(label_set)
