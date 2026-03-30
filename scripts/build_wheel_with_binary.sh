#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BIN_DIR="$ROOT_DIR/dbt_vectorize/bin"
mkdir -p "$BIN_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

cleanup() {
  rm -f "$BIN_DIR/pg_embedder" "$BIN_DIR/pg_embedder.exe"
  rm -f "$BIN_DIR/pg_search" "$BIN_DIR/pg_search.exe"
}
trap cleanup EXIT

echo "[1/4] Building Rust backend (release)..."
cargo build --manifest-path rust/embedding_engine/Cargo.toml --release --bin pg_embedder --bin pg_search

copy_bin() {
  local name="$1"
  if [[ -f "$ROOT_DIR/rust/embedding_engine/target/release/${name}" ]]; then
    cp "$ROOT_DIR/rust/embedding_engine/target/release/${name}" "$BIN_DIR/${name}"
    chmod +x "$BIN_DIR/${name}"
  elif [[ -f "$ROOT_DIR/rust/embedding_engine/target/release/${name}.exe" ]]; then
    cp "$ROOT_DIR/rust/embedding_engine/target/release/${name}.exe" "$BIN_DIR/${name}.exe"
  else
    echo "Could not find built ${name} binary in rust/embedding_engine/target/release" >&2
    exit 1
  fi
}

copy_bin pg_embedder
copy_bin pg_search

echo "[2/4] Cleaning previous dist artifacts..."
rm -rf dist/

echo "[3/4] Building Python wheel/sdist..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel build
"$PYTHON_BIN" -m build --no-isolation

echo "[4/4] Done. Artifacts in dist/"
ls -lh dist/
