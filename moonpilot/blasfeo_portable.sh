#!/usr/bin/env bash
# Dev-PC only: replace comma-deps-acados' libblasfeo.so with a portable build.
#
# The x86 wheel builds blasfeo with BLASFEO_TARGET=X64_AUTOMATIC, i.e. for the CI host (Haswell:
# AVX2 + FMA), so the longitudinal MPC dies with SIGILL on CPUs without AVX2. blasfeo's own
# X64_INTEL_CORE (SSE3) kernels at this revision still contain AVX, so this builds GENERIC (plain C)
# from the exact blasfeo the wheel pins. Double-precision panel size is 4 on both targets, so the
# hpipm/acados binaries compiled against the wheel's headers stay compatible.
#
# A reinstall or version bump of comma-deps-acados restores the AVX2 library; rerun this afterwards.
# Usage: moonpilot/blasfeo_portable.sh [--force]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACADOS_COMMIT="8af9b0ad180940ef611884574a0b27a43504311d"  # commaai/dependencies acados/build.sh (v0.2.2)

if [ "${1:-}" != "--force" ] && grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; then
  echo "CPU has AVX2+FMA; the wheel's library runs here. Pass --force to rebuild anyway."
  exit 0
fi

LIB="$("$ROOT/.venv/bin/python" -c 'import acados; print(acados.LIB_DIR)')"
INC="$("$ROOT/.venv/bin/python" -c 'import acados; print(acados.INCLUDE_DIR)')/blasfeo/include"
GEN="$ROOT/openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git init -q "$WORK/acados"
git -C "$WORK/acados" fetch -q --depth 1 https://github.com/acados/acados.git "$ACADOS_COMMIT"
git -C "$WORK/acados" checkout -q FETCH_HEAD
git -C "$WORK/acados" submodule update -q --init --depth 1 external/blasfeo
SRC="$WORK/acados/external/blasfeo"

# The source must be the one the installed headers came from; anything else is a different ABI.
if ! diff -rq -x blasfeo_target.h "$SRC/include" "$INC" >/dev/null; then
  echo "installed blasfeo headers differ from acados $ACADOS_COMMIT; update ACADOS_COMMIT" >&2
  exit 1
fi

make -s -C "$SRC" shared_library TARGET=GENERIC BLAS_API=0 -j"$(nproc)" >/dev/null
OUT="$SRC/lib/libblasfeo.so"

if objdump -d --no-show-raw-insn "$OUT" | grep -qE '\sv[a-z]+'; then
  echo "built library still contains VEX (AVX) instructions" >&2
  exit 1
fi

cp -n "$LIB/libblasfeo.so" "$LIB/libblasfeo.so.haswell"
cp "$OUT" "$LIB/libblasfeo.so"
if [ -d "$GEN" ]; then
  cp "$OUT" "$GEN/libblasfeo.so"
fi
echo "installed portable libblasfeo.so (original kept as $LIB/libblasfeo.so.haswell)"
