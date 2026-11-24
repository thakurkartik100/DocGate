"""Build the document corpus: one chunk per public function/class/method in selected stdlib modules.

The corpus is snapshotted to data/corpus.jsonl so results do not depend on the Python version
that happens to run the eval. Rebuild with: python -m docgate.corpus
"""
import importlib
import inspect
import json
from pathlib import Path

MODULES = [
    "json", "re", "itertools", "collections", "functools", "pathlib", "datetime", "subprocess",
    "logging", "argparse", "heapq", "bisect", "shutil", "csv", "sqlite3", "urllib.parse",
    "hashlib", "random", "statistics", "textwrap", "tempfile", "threading", "queue", "zipfile",
    "gzip", "time", "math", "copy", "enum", "dataclasses", "os.path", "string", "operator",
    "contextlib", "glob", "fnmatch", "pickle", "base64", "uuid", "secrets", "decimal", "fractions",
    "asyncio", "concurrent.futures", "socket", "struct", "zlib", "tarfile", "configparser", "difflib",
    "pprint", "traceback", "warnings", "weakref", "abc", "typing", "unittest.mock", "inspect",
    "calendar", "shlex", "html", "ipaddress", "smtplib", "email.message", "http.client", "signal",
]
_NOISE = set(dir(BaseException)) | set(dir(object))  # inherited exception/object plumbing
MAX_DOC = 1200
MIN_DOC = 15
OUT = Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"


def _chunk(mod_name: str, qualname: str, obj, context: str = "") -> dict | None:
    doc = inspect.getdoc(obj)
    if not doc or len(doc) < MIN_DOC:
        return None
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = ""
    cid = f"{mod_name}.{qualname}"
    return {"id": cid, "module": mod_name, "text": f"{context}{cid}{sig}\n{doc[:MAX_DOC]}"}


def build(context: bool = True) -> list[dict]:
    """context=True prefixes each chunk with its module summary line (cheap contextual retrieval)."""
    chunks, seen = [], set()
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        public = set(getattr(mod, "__all__", ()))
        top = mod_name.split(".")[0]
        summary = (inspect.getdoc(mod) or "").split("\n")[0]
        ctx = f"[{mod_name}: {summary}]\n" if context and summary else ""
        for name, obj in inspect.getmembers(mod):
            # C-accelerated and re-exported names (json.dumps, heapq.heappush) live in other modules,
            # so accept names in __all__ or whose defining module shares this package's top-level name.
            owned = name in public or (getattr(obj, "__module__", None) or "").split(".")[0] in (top, f"_{top}")
            if name.startswith("_") or not owned:
                continue
            targets = []
            if inspect.isroutine(obj):
                targets.append((name, obj))
            elif inspect.isclass(obj):
                targets.append((name, obj))
                for m_name, m in inspect.getmembers(obj, callable):
                    # skip methods inherited from builtins/other packages (e.g. StrEnum.split is str.split)
                    owner = next((b for b in obj.__mro__ if m_name in vars(b)), None)  # None: metaclass attribute
                    if owner and not m_name.startswith("_") and m_name not in _NOISE and owner.__module__.split(".")[0] in (top, f"_{top}"):
                        targets.append((f"{name}.{m_name}", m))
            for qn, o in targets:
                c = _chunk(mod_name, qn, o, ctx)
                if c and c["id"] not in seen:
                    seen.add(c["id"])
                    chunks.append(c)
    return chunks


def load(path: Path = OUT) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    import sys

    plain = "--no-context" in sys.argv  # ablation baseline: data/corpus_plain.jsonl
    data = build(context=not plain)
    dest = OUT.with_name("corpus_plain.jsonl") if plain else OUT
    OUT.parent.mkdir(exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for c in data:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(data)} chunks from {len(MODULES)} modules -> {dest}")
