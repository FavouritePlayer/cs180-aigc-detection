"""Download the Frontier manifest's 100 original images from their pinned HF dataset revision."""
import argparse
import hashlib
import time
from pathlib import Path

import requests

BASE = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{file}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import json
    rows = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    for row in rows:
        dest = out / row["file"]
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == row["sha256"]:
            print("CACHED", row["id"], flush=True)
            continue
        url = BASE.format(dataset=row["dataset"], revision=row["revision"], file=row["file"])
        for attempt in range(3):
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                break
            time.sleep(2)
        else:
            print("FAILED", row["id"], resp.status_code, flush=True)
            continue
        digest = hashlib.sha256(resp.content).hexdigest()
        assert digest == row["sha256"], f"hash mismatch for {row['id']}: {digest} != {row['sha256']}"
        dest.write_bytes(resp.content)
        print("OK", row["id"], flush=True)


if __name__ == "__main__":
    main()
