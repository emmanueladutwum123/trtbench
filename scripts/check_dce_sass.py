#!/usr/bin/env python3
"""Prove from the compiled SASS that trap T1's kernel really was eliminated.

The T1 trap claims that an FMA loop whose result is never stored gets deleted
by the compiler. This reads the SASS of every architecture in the binary and
checks: trap_dead_fma_kernel contains no FFMA, while the live fma_kernel (the
control, same arithmetic, result stored) does. Needs cuobjdump, not a GPU.

    python3 scripts/check_dce_sass.py build/trtbench
"""
import re
import subprocess
import sys
from collections import defaultdict

DEAD = re.compile(r"20trap_dead_fma_kernel")
LIVE = re.compile(r"(?<![A-Za-z0-9_])10fma_kernel")


def functions(sass: str):
    """Yield (arch, mangled name, instruction lines) for each function."""
    arch, name, body = None, None, []
    for line in sass.splitlines():
        m = re.search(r"code for (sm_\d+)", line)
        if m:
            if name:
                yield arch, name, body
            arch, name, body = m.group(1), None, []
            continue
        m = re.search(r"Function : (\S+)", line)
        if m:
            if name:
                yield arch, name, body
            name, body = m.group(1), []
            continue
        if name and re.search(r"/\*[0-9a-f]{4,}\*/", line):
            body.append(line)
    if name:
        yield arch, name, body


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else "build/trtbench"
    sass = subprocess.run(["cuobjdump", "-sass", binary], check=True,
                          capture_output=True, text=True).stdout
    ffma = defaultdict(dict)  # kind -> arch -> count
    for arch, name, body in functions(sass):
        kind = "dead" if DEAD.search(name) else "live" if LIVE.search(name) else None
        if kind:
            ffma[kind][arch] = sum(1 for l in body if re.search(r"\bFFMA\b", l))
            if kind == "dead":
                print(f"{arch} trap_dead_fma_kernel: {len(body)} instructions, "
                      f"{ffma[kind][arch]} FFMA")
            else:
                print(f"{arch} fma_kernel (control): {ffma[kind][arch]} FFMA")

    ok = True
    if not ffma["dead"] or not ffma["live"]:
        print("FAIL: kernels not found in SASS", file=sys.stderr)
        return 1
    for arch, n in ffma["dead"].items():
        if n:
            print(f"FAIL: {arch} dead kernel still contains {n} FFMA", file=sys.stderr)
            ok = False
    for arch, n in ffma["live"].items():
        if not n:
            print(f"FAIL: {arch} live control has no FFMA; the check proves nothing",
                  file=sys.stderr)
            ok = False
    print("PASS: dead-code elimination confirmed in SASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
