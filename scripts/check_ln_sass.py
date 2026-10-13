#!/usr/bin/env python3
"""Check from the compiled SASS that each Phase 4 kernel does what its
description in include/trtbench/cuda/layernorm.hpp claims. Needs cuobjdump,
not a GPU; runs in CI on every architecture in the binary.

  warp   128-bit global loads and stores, warp shuffles (SHFL.BFLY), and no
         shared memory or block barrier
  block  shared memory and BAR.SYNC (the tree reduction), no 128-bit loads
  naive  no 128-bit loads, no shared memory, no shuffles

    python3 scripts/check_ln_sass.py build/trtbench
"""
import re
import subprocess
import sys
from collections import defaultdict

from check_dce_sass import functions

PATTERNS = {
    "ldg128": r"\bLDG\.E\S*\.128\b",
    "stg128": r"\bSTG\.E\S*\.128\b",
    "shfl": r"\bSHFL\.BFLY\b",
    "bar": r"\bBAR\.SYNC",
    "shared": r"\b(?:LDS|STS)\b",
}
RULES = {  # kernel -> {feature: must be present?}
    "add_ln_warp": {"ldg128": True, "stg128": True, "shfl": True, "bar": False, "shared": False},
    "add_ln_block": {"ldg128": False, "bar": True, "shared": True},
    "add_ln_naive": {"ldg128": False, "shared": False, "shfl": False},
}


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else "build/trtbench"
    sass = subprocess.run(["cuobjdump", "-sass", binary], check=True,
                          capture_output=True, text=True).stdout
    seen = defaultdict(int)
    ok = True
    for arch, name, body in functions(sass):
        kernel = next((k for k in RULES if k in name), None)
        if kernel is None:
            continue
        seen[kernel] += 1
        text = "\n".join(body)
        found = {f: len(re.findall(p, text)) for f, p in PATTERNS.items()}
        bad = [f"{f} {'missing' if want else 'present'}" for f, want in RULES[kernel].items()
               if bool(found[f]) != want]
        summary = " ".join(f"{f}={n}" for f, n in found.items())
        print(f"{arch} {kernel:13s} {'FAIL ' + ', '.join(bad) if bad else 'ok':28s} {summary}  {name[:60]}")
        ok &= not bad
    for k in RULES:
        if not seen[k]:
            print(f"FAIL: {k} not found in SASS", file=sys.stderr)
            ok = False
    print("PASS: SASS matches every kernel's description" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
