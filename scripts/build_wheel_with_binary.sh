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
}
trap cleanup EXIT

echo "[1/4] Building Rust backend (release)..."
cargo build --manifest-path rust/embedding_engine/Cargo.toml --release --bin pg_embedder

if [[ -f "$ROOT_DIR/rust/embedding_engine/target/release/pg_embedder" ]]; then
  cp "$ROOT_DIR/rust/embedding_engine/target/release/pg_embedder" "$BIN_DIR/pg_embedder"
  chmod +x "$BIN_DIR/pg_embedder"
elif [[ -f "$ROOT_DIR/rust/embedding_engine/target/release/pg_embedder.exe" ]]; then
  cp "$ROOT_DIR/rust/embedding_engine/target/release/pg_embedder.exe" "$BIN_DIR/pg_embedder.exe"
else
  echo "Could not find built pg_embedder binary in rust/embedding_engine/target/release" >&2
  exit 1
fi

echo "[2/4] Cleaning previous dist artifacts..."
rm -rf dist/

echo "[3/4] Building Python wheel/sdist..."
"$PYTHON_BIN" -m build --no-isolation

echo "[4/4] Done. Artifacts in dist/"
ls -lh dist/
