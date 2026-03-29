from __future__ import annotations

import os
import shutil
import subprocess
import sys
import json
from pathlib import Path

import yaml

def _resolve_cwd() -> Path:
    explicit = os.environ.get("DBT_VECTORIZE_CWD")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path.cwd()


def _find_repo_root(start: Path) -> Path | None:
    env_root = os.environ.get("DBT_VECTORIZE_REPO")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "dbt_project.yml").exists() and (p / "rust" / "embedding_engine" / "Cargo.toml").exists():
            return p

    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "dbt_project.yml").exists() and (p / "rust" / "embedding_engine" / "Cargo.toml").exists():
            return p
    return None


def _packaged_embedder() -> str | None:
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "bin" / "pg_embedder",
        pkg_dir / "bin" / "pg_embedder.exe",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


def _find_pg_embedder_cmd(cwd: Path) -> tuple[list[str], Path | None]:
    explicit = os.environ.get("DBT_VECTORIZE_PG_EMBEDDER")
    if explicit:
        return [explicit], cwd

    packaged = _packaged_embedder()
    if packaged:
        return [packaged], cwd

    cargo = shutil.which("cargo")
    repo = _find_repo_root(cwd)
    if cargo and repo:
        return [cargo, "run", "--quiet", "--bin", "pg_embedder", "--release", "--"], repo / "rust" / "embedding_engine"

    raise FileNotFoundError(
        "Could not find runnable pg_embedder backend. "
        "Set DBT_VECTORIZE_PG_EMBEDDER, install wheel with bundled binary, "
        "or run inside a cloned dbt-vector repo with Rust/cargo available."
    )


def _build_dbt_cmd(cwd: Path, argv: list[str]) -> tuple[list[str], dict[str, str], str, str]:
    dbt = os.environ.get("DBT", "dbt")
    profile_dir = os.environ.get("PROFILE_DIR") or os.environ.get("DBT_PROFILES_DIR") or str(cwd)
    project_dir = os.environ.get("PROJECT_DIR") or str(cwd)
    select_model = os.environ.get("SELECT_MODEL", "vector_knowledge_base")
    embed_provider = os.environ.get("EMBED_PROVIDER", "local")
    embed_model = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    run_args = list(argv) if argv else ["--select", select_model]
    cmd = [
        dbt,
        "run",
        "--no-partial-parse",
        "--profiles-dir",
        profile_dir,
        "--project-dir",
        project_dir,
        *run_args,
    ]
    env = os.environ.copy()
    env["EMBED_PROVIDER"] = embed_provider
    env["EMBED_MODEL"] = embed_model
    return cmd, env, embed_provider, embed_model


def _selected_model_from_args(argv: list[str], default: str) -> str:
    if not argv:
        return default
    model = default
    for i, arg in enumerate(argv):
        if arg in ("--select", "-s") and i + 1 < len(argv):
            model = argv[i + 1]
    return model.split(".")[-1]


def _infer_target_from_manifest(project_dir: Path, model_name: str, env: dict[str, str]) -> None:
    manifest = project_dir / "target" / "manifest.json"
    if not manifest.exists():
        return
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return
    nodes = (data.get("nodes") or {}).values()
    node = next(
        (n for n in nodes if n.get("resource_type") == "model" and n.get("name") == model_name),
        None,
    )
    if not node:
        return
    cfg = node.get("config") or {}
    index_name = cfg.get("index_name") or node.get("alias") or node.get("name")
    schema = node.get("schema")
    if env.get("INDEX_NAME", "knowledge_base") == "knowledge_base" and index_name:
        env["INDEX_NAME"] = str(index_name)
    if env.get("SCHEMA", "public") == "public" and schema:
        env["SCHEMA"] = str(schema)


def _build_embed_env(cwd: Path) -> dict[str, str]:
    env = os.environ.copy()
    profile_dir = env.get("PROFILE_DIR") or env.get("DBT_PROFILES_DIR") or str(cwd)
    profile_name = env.get("PROFILE", "default")
    target_name = env.get("TARGET")
    profile_file = env.get("PROFILE_FILE") or str(Path(profile_dir) / "profiles.yml")

    # Fallback to dbt profile values when PG* are not explicitly provided.
    if (
        not env.get("PGHOST")
        or not env.get("PGPORT")
        or not env.get("PGUSER")
        or not env.get("PGDATABASE")
    ):
        p = Path(profile_file)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            profile = data.get(profile_name, {}) or {}
            outputs = profile.get("outputs", {}) or {}
            target = target_name or profile.get("target")
            if not target and outputs:
                target = next(iter(outputs.keys()))
            cfg = outputs.get(target, {}) if target else {}
            if cfg.get("type") == "postgres":
                if not env.get("PGHOST") and cfg.get("host") is not None:
                    env["PGHOST"] = str(cfg["host"])
                if not env.get("PGPORT") and cfg.get("port") is not None:
                    env["PGPORT"] = str(cfg["port"])
                if not env.get("PGUSER") and cfg.get("user") is not None:
                    env["PGUSER"] = str(cfg["user"])
                if not env.get("PGPASSWORD") and cfg.get("password") is not None:
                    env["PGPASSWORD"] = str(cfg["password"])
                if not env.get("PGDATABASE") and cfg.get("dbname") is not None:
                    env["PGDATABASE"] = str(cfg["dbname"])
                if not env.get("SCHEMA") and cfg.get("schema") is not None:
                    env["SCHEMA"] = str(cfg["schema"])

    env.setdefault("PGHOST", "localhost")
    env.setdefault("PGPORT", "5432")
    env.setdefault("PGUSER", "postgres")
    env.setdefault("PGPASSWORD", "")
    env.setdefault("PGDATABASE", "postgres")
    env.setdefault("SCHEMA", "public")
    env.setdefault("INDEX_NAME", "knowledge_base")
    env.setdefault("EMBED_DIMS", "1536")
    env.setdefault("EMBED_PROVIDER", "local")
    env.setdefault("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return env


def main() -> int:
    argv = sys.argv[1:]
    cwd = _resolve_cwd()
    select_model = os.environ.get("SELECT_MODEL", "vector_knowledge_base")
    selected_model = _selected_model_from_args(argv, select_model)

    dbt_cmd, dbt_env, provider, model = _build_dbt_cmd(cwd, argv)
    print(f"[vectorize] running dbt model {selected_model} (provider={provider}, model={model})")
    dbt_proc = subprocess.run(dbt_cmd, cwd=str(cwd), env=dbt_env)
    if dbt_proc.returncode != 0:
        return dbt_proc.returncode

    embed_cmd, embed_cwd = _find_pg_embedder_cmd(cwd)
    embed_env = _build_embed_env(cwd)
    project_dir = Path(os.environ.get("PROJECT_DIR", str(cwd))).expanduser().resolve()
    _infer_target_from_manifest(project_dir, selected_model, embed_env)
    schema = embed_env.get("SCHEMA", "public")
    index_name = embed_env.get("INDEX_NAME", "knowledge_base")
    print(f"[vectorize] generating embeddings via Rust into {schema}.{index_name}")
    embed_proc = subprocess.run(embed_cmd, cwd=str(embed_cwd or cwd), env=embed_env)
    if embed_proc.returncode != 0:
        return embed_proc.returncode

    print("[vectorize] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
