"""Write a native-eval runtime manifest that pins one method's adapter sources and checkpoint.

  python make_runtime_manifest.py --method dflash --checkpoint <ckpt_final.pt> --out <manifest.json>

Paths are absolute; SHA256 values are computed from the files as they are now.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = ["contract.py", "spatial_adapter.py"]
# method -> (bootstrap file, extra adapter sources, implementation scope)
METHODS = {
    "dflash": ("bootstrap_dflash.py", [],
               "Qwen3.6-35B-A3B trained DFlash (400K, 75000 updates), interleaved partial spatial MRoPE, "
               "target embedding/head shared, official vLLM linear verifier"),
    "dfly": ("bootstrap_dfly.py", ["dfly_adapter.py"],
             "Qwen3.6-35B-A3B trained DFly (400K, 75000 updates, upstream two-stage recipe): FC base + "
             "per-layer fusion residual context, sequential greedy hidden correction with real previous tokens, "
             "spatial MRoPE, official vLLM linear verifier"),
    "dflare": ("bootstrap_dflare.py", ["dflare_adapter.py"],
               "Qwen3.6-35B-A3B trained DFlare (400K, 75000 updates): per-layer softmax fusion of target "
               "layers [1,19,37] with dedicated context K/V projections (no FC), spatial MRoPE, stock greedy "
               "head, official vLLM linear verifier"),
    "jetspec": ("bootstrap_jetspec.py", ["jetspec_adapter.py"],
                "Qwen3.6-35B-A3B trained JetSpec (400K, 75000 updates): local shared-block integrated single-path "
                "reference with block-causal draft queries (not the upstream parallel-tree runtime), spatial MRoPE, "
                "official vLLM linear verifier"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bootstrap, extra, scope = METHODS[args.method]
    sources = [HERE / name for name in (bootstrap, *COMMON, *extra)]
    checkpoint = args.checkpoint.resolve()
    manifest = {
        "method": args.method, "scope": scope,
        "bootstrap": str(HERE / bootstrap), "checkpoint": str(checkpoint), "environment": {},
        "files": {str(path): sha256(path) for path in (*sources, checkpoint)},
    }
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
