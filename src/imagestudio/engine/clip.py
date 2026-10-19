"""Semantic gallery search with CLIP ViT-B/32 (OpenAI, MIT) on onnxruntime: every finished
image gets a 512-d embedding once; a search sentence is embedded the same way and the gallery
is ranked by cosine similarity. The same embeddings find near-duplicates. Everything runs on
this machine; the two quantised models total about 150 MB.

The tokenizer is CLIP's byte-level BPE, implemented here from vocab.json and merges.txt so no
extra dependency is needed."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from imagestudio.engine.base import EngineError

IMAGE_SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
CONTEXT = 77
DIM = 512


def _ensure_runtime():
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise EngineError(
            "semantic search needs onnxruntime, numpy and Pillow: "
            "pip install 'presidium-imagestudio[bg]'"
        ) from e
    return np, ort, Image


# ---------------------------------------------------------------- tokenizer


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


class ClipTokenizer:
    """CLIP's lower-cased byte-level BPE with <|startoftext|> / <|endoftext|>, padded to 77."""

    _pat = re.compile(
        r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[\d]|[^\s\w\d]+",
        re.IGNORECASE,
    )

    def __init__(self, vocab_path: Path, merges_path: Path) -> None:
        self.encoder: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        merges = merges_path.read_text(encoding="utf-8").split("\n")
        merges = [m for m in merges[1:] if m.strip() and not m.startswith("#version")]
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.cache: dict[str, str] = {}
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = (*tuple(token[:-1]), token[-1] + "</w>")
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        text = " ".join(text.strip().lower().split())
        ids: list[int] = []
        for tok in self._pat.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self._bpe(tok).split(" ") if t in self.encoder)
        ids = [self.sot, *ids[: CONTEXT - 2], self.eot]
        return ids

    def batch(self, text: str):
        """input_ids and attention_mask as int64 arrays of shape (1, 77)."""
        np, _ort, _img = _ensure_runtime()
        ids = self.encode(text)
        input_ids = np.full((1, CONTEXT), self.eot, dtype=np.int64)
        mask = np.zeros((1, CONTEXT), dtype=np.int64)
        input_ids[0, : len(ids)] = ids
        mask[0, : len(ids)] = 1
        return input_ids, mask


# ---------------------------------------------------------------- embeddings


class ClipEngine:
    """Image and text embeddings, unit length, from the two ONNX halves of CLIP."""

    name = "onnxruntime"

    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}
        self._tokenizer: ClipTokenizer | None = None

    def available(self) -> bool:
        try:
            _ensure_runtime()
        except EngineError:
            return False
        return True

    def _session(self, model_path: Path):
        key = str(model_path)
        if key not in self._sessions:
            _np, ort, _img = _ensure_runtime()
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._sessions[key] = ort.InferenceSession(
                key, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return self._sessions[key]

    def tokenizer(self, vocab: Path, merges: Path) -> ClipTokenizer:
        if self._tokenizer is None:
            self._tokenizer = ClipTokenizer(vocab, merges)
        return self._tokenizer

    def embed_image(self, image: Path, vision_model: Path):
        np, _ort, image_mod = _ensure_runtime()
        sess = self._session(vision_model)
        with image_mod.open(image) as src:
            im = src.convert("RGB")
            w, h = im.size
            scale = IMAGE_SIZE / min(w, h)
            im = im.resize(
                (max(IMAGE_SIZE, round(w * scale)), max(IMAGE_SIZE, round(h * scale))),
                image_mod.BICUBIC,
            )
            left = (im.width - IMAGE_SIZE) // 2
            top = (im.height - IMAGE_SIZE) // 2
            im = im.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        x = x.transpose(2, 0, 1)[None]
        out = sess.run(None, {sess.get_inputs()[0].name: x})
        return _unit(np, _pick_embedding(out, sess))

    def embed_text(self, text: str, text_model: Path, vocab: Path, merges: Path):
        np, _ort, _img = _ensure_runtime()
        sess = self._session(text_model)
        input_ids, mask = self.tokenizer(vocab, merges).batch(text)
        feed = {}
        for inp in sess.get_inputs():
            if "mask" in inp.name:
                feed[inp.name] = mask
            else:
                feed[inp.name] = input_ids
        out = sess.run(None, feed)
        return _unit(np, _pick_embedding(out, sess))


def _pick_embedding(outputs, sess):
    """The exported halves return (last_hidden_state, pooled/projected embeds); take the
    output named like an embedding, else the first 2-d one of the right width."""
    names = [o.name for o in sess.get_outputs()]
    for name, arr in zip(names, outputs, strict=True):
        if "embeds" in name or "pooler" in name:
            return arr[0]
    for arr in outputs:
        if arr.ndim == 2 and arr.shape[1] == DIM:
            return arr[0]
    raise EngineError(f"unexpected CLIP outputs: {names}")


def _unit(np, v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


# ---------------------------------------------------------------- index


class EmbeddingIndex:
    """One .npz next to the jobs: ids and a (n, 512) float32 matrix. Small enough to hold in
    memory for tens of thousands of images; rewritten atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.ids: list[str] = []
        self.vectors = None
        self._load()

    def _load(self) -> None:
        try:
            np, _ort, _img = _ensure_runtime()
        except EngineError:
            return
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as z:
                self.ids = [str(x) for x in z["ids"].tolist()]
                self.vectors = z["vectors"].astype(np.float32)
        except Exception:
            self.ids, self.vectors = [], None

    def save(self) -> None:
        np, _ort, _img = _ensure_runtime()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        vectors = self.vectors if self.vectors is not None else np.zeros((0, DIM), np.float32)
        np.savez(tmp, ids=np.array(self.ids), vectors=vectors)
        tmp.replace(self.path)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self.ids

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, job_id: str, vector) -> None:
        np, _ort, _img = _ensure_runtime()
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if job_id in self.ids:
            self.vectors[self.ids.index(job_id)] = v[0]
            return
        self.ids.append(job_id)
        self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])

    def remove(self, job_id: str) -> None:
        if job_id not in self.ids:
            return
        np, _ort, _img = _ensure_runtime()
        i = self.ids.index(job_id)
        self.ids.pop(i)
        self.vectors = np.delete(self.vectors, i, axis=0)

    def search(self, vector, limit: int = 50) -> list[tuple[str, float]]:
        """(job_id, cosine similarity) best first."""
        if self.vectors is None or not len(self.ids):
            return []
        np, _ort, _img = _ensure_runtime()
        sims = self.vectors @ np.asarray(vector, dtype=np.float32).reshape(-1)
        order = np.argsort(-sims)[:limit]
        return [(self.ids[i], float(sims[i])) for i in order]

    def duplicates(self, threshold: float = 0.96) -> list[list[str]]:
        """Groups of near-identical images (cosine above the threshold), largest first."""
        if self.vectors is None or len(self.ids) < 2:
            return []
        _np, _ort, _img = _ensure_runtime()
        sims = self.vectors @ self.vectors.T
        n = len(self.ids)
        seen: set[int] = set()
        groups: list[list[str]] = []
        for i in range(n):
            if i in seen:
                continue
            close = [j for j in range(n) if j != i and sims[i, j] >= threshold and j not in seen]
            if close:
                group = [i, *close]
                seen.update(group)
                groups.append([self.ids[k] for k in group])
        groups.sort(key=len, reverse=True)
        return groups
