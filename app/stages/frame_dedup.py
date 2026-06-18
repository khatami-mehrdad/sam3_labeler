"""Perceptual-hash dedup. Within each `origin` (each source folder / each video),
collapse near-duplicates using pHash + dHash agreement.

Cheap, runs CPU-only at ~500 imgs/sec.
"""
import concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

HAMMING_THRESH = 6   # ≤ this many bits different = considered a duplicate
N_WORKERS = 16


def _hash_one(p: str):
    try:
        with Image.open(p) as im:
            return str(imagehash.phash(im, hash_size=8)), str(imagehash.dhash(im, hash_size=8))
    except Exception:
        return None, None


def phash_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Add phash/dhash + cluster_id columns and return only one rep per cluster.

    Dedup is scoped per `origin` (so multiple videos don't get cross-collapsed).
    """
    if len(df) == 0:
        return df

    paths = df["path"].tolist()
    ph = [None] * len(paths)
    dh = [None] * len(paths)
    with cf.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_hash_one, p): i for i, p in enumerate(paths)}
        for fut in tqdm(cf.as_completed(futs), total=len(futs),
                         desc="phash", leave=False):
            i = futs[fut]
            ph[i], dh[i] = fut.result()
    df = df.copy()
    df["phash"] = ph
    df["dhash"] = dh
    df = df.dropna(subset=["phash", "dhash"]).reset_index(drop=True)

    # Bucket by first 4 hex chars (16 bits) of phash — same bucket = candidates
    out_parts = []
    for origin, sub in df.groupby("origin"):
        sub = sub.reset_index(drop=True)
        buckets: dict[str, list[tuple[int, imagehash.ImageHash]]] = defaultdict(list)
        for i, row in sub.iterrows():
            ih = imagehash.hex_to_hash(row["phash"])
            buckets[row["phash"][:4]].append((i, ih))

        cluster_of = [-1] * len(sub)
        next_cid = 0
        for items in buckets.values():
            for idx, ih in items:
                if cluster_of[idx] >= 0:
                    continue
                cluster_of[idx] = next_cid
                for idx2, ih2 in items:
                    if idx2 == idx or cluster_of[idx2] >= 0:
                        continue
                    if (ih - ih2) <= HAMMING_THRESH:
                        cluster_of[idx2] = next_cid
                next_cid += 1

        sub["cluster_id"] = [f"{origin}::c{c}" for c in cluster_of]
        sub["wh"] = sub["width"] * sub["height"]
        # Keep largest image per cluster (more info)
        reps = (sub.sort_values("wh", ascending=False)
                  .drop_duplicates("cluster_id")
                  .drop(columns=["wh"])
                  .reset_index(drop=True))
        out_parts.append(reps)

    return pd.concat(out_parts, ignore_index=True)
