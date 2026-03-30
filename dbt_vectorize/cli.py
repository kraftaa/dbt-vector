from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from dbt_vectorize import __version__

SUBCOMMANDS = {"build", "embed", "search", "inspect", "init"}


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


def _packaged_binary(binary_name: str) -> str | None:
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "bin" / binary_name,
        pkg_dir / "bin" / f"{binary_name}.exe",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


def _find_binary_cmd(binary_name: str, cwd: Path) -> tuple[list[str], Path | None]:
    env_override = {
        "pg_embedder": "DBT_VECTORIZE_PG_EMBEDDER",
        "pg_search": "DBT_VECTORIZE_PG_SEARCH",
    }.get(binary_name)
    if env_override and os.environ.get(env_override):
        return [os.environ[env_override]], cwd

    packaged = _packaged_binary(binary_name)
    if packaged:
        return [packaged], cwd

    cargo = shutil.which("cargo")
    repo = _find_repo_root(cwd)
    if cargo and repo:
        return [
            cargo,
            "run",
            "--quiet",
            "--bin",
            binary_name,
            "--release",
            "--",
        ], repo / "rust" / "embedding_engine"

    raise FileNotFoundError(
        f"Could not find runnable {binary_name} backend. "
        f"Set DBT_VECTORIZE_{binary_name.upper()} or run in a cloned dbt-vector repo with cargo."
    )


def _dbt_flags(profiles_dir: str, project_dir: str, profile_name: str | None, target_name: str | None) -> list[str]:
    out = ["--profiles-dir", profiles_dir, "--project-dir", project_dir]
    if profile_name:
        out.extend(["--profile", profile_name])
    if target_name:
        out.extend(["--target", target_name])
    return out


def _resolve_model_name(
    dbt_cmd: str,
    selector: str,
    profiles_dir: str,
    project_dir: str,
    profile_name: str | None,
    target_name: str | None,
    env: dict[str, str],
    cwd: Path,
) -> str:
    cmd = [
        dbt_cmd,
        "ls",
        "--resource-type",
        "model",
        "--output",
        "json",
        "--select",
        selector,
        *_dbt_flags(profiles_dir, project_dir, profile_name, target_name),
    ]
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "dbt ls failed")
    models: list[str] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("resource_type") != "model":
            continue
        name = payload.get("name")
        if isinstance(name, str) and name:
            models.append(name)

    models = sorted(set(models))
    if not models:
        raise RuntimeError(f"Selector '{selector}' did not resolve to any model.")
    if len(models) > 1:
        raise RuntimeError(
            f"Selector '{selector}' resolved to multiple models ({', '.join(models)}). "
            "Please pass one model selector."
        )
    return models[0].split(".")[-1]


def _validated_columns_arg(columns_arg: str | None) -> str | None:
    if not columns_arg:
        return None
    safe_cols: list[str] = []
    for col in columns_arg.split(","):
        c = col.strip()
        if not c:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c):
            raise RuntimeError(
                f"Invalid column '{c}' in --columns. Use comma-separated SQL identifiers only."
            )
        safe_cols.append(c)
    if not safe_cols:
        return None
    return ",".join(safe_cols)


def _normalize_git_url(url: str) -> str:
    out = url.strip()
    if out.endswith(".git"):
        out = out[:-4]
    return out.rstrip("/")


def _packages_install_path(project_dir: Path) -> Path:
    project_file = project_dir / "dbt_project.yml"
    if not project_file.exists():
        return project_dir / "dbt_packages"
    loaded = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return project_dir / "dbt_packages"
    rel = loaded.get("packages-install-path")
    if isinstance(rel, str) and rel.strip():
        return (project_dir / rel.strip()).resolve()
    return project_dir / "dbt_packages"


def _has_vector_index_materialization(project_dir: Path) -> bool:
    local_macro = project_dir / "macros" / "materializations" / "vector_index.sql"
    if local_macro.exists():
        return True
    pkg_root = _packages_install_path(project_dir)
    if not pkg_root.exists():
        return False
    for candidate in pkg_root.glob("*/macros/materializations/vector_index.sql"):
        if candidate.exists():
            return True
    return False


def _ensure_package_entry(project_dir: Path, git_url: str, revision: str) -> tuple[Path, bool]:
    packages_file = project_dir / "packages.yml"
    existing_data: dict = {}
    if packages_file.exists():
        loaded = yaml.safe_load(packages_file.read_text(encoding="utf-8"))
        if loaded is None:
            existing_data = {}
        elif isinstance(loaded, dict):
            existing_data = loaded
        else:
            raise RuntimeError("packages.yml must contain a YAML mapping at top level.")

    packages = existing_data.get("packages")
    if packages is None:
        packages = []
    if not isinstance(packages, list):
        raise RuntimeError("packages.yml key 'packages' must be a list.")

    target_norm = _normalize_git_url(git_url)
    changed = False
    found = False

    for item in packages:
        if not isinstance(item, dict):
            continue
        raw_git = item.get("git")
        if not isinstance(raw_git, str):
            continue
        if _normalize_git_url(raw_git) != target_norm:
            continue
        found = True
        if item.get("revision") != revision:
            item["revision"] = revision
            changed = True
        break

    if not found:
        packages.append({"git": git_url, "revision": revision})
        changed = True

    if changed or "packages" not in existing_data:
        existing_data["packages"] = packages
        packages_file.write_text(
            yaml.safe_dump(existing_data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )

    return packages_file, changed


def _load_manifest_node(project_dir: Path, model_name: str) -> dict | None:
    manifest = project_dir / "target" / "manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    nodes = (data.get("nodes") or {}).values()
    return next(
        (
            n
            for n in nodes
            if n.get("resource_type") == "model" and n.get("name") == model_name
        ),
        None,
    )


def _resolve_index_schema_from_node(node: dict | None) -> tuple[str | None, str | None]:
    if not node:
        return None, None
    cfg = node.get("config") or {}
    index_name = cfg.get("index_name") or node.get("alias") or node.get("name")
    schema = node.get("schema")
    return (str(index_name) if index_name else None, str(schema) if schema else None)


def _apply_manifest_target(
    project_dir: Path,
    model_name: str,
    env: dict[str, str],
    *,
    set_index: bool,
    set_schema: bool,
) -> tuple[bool, bool]:
    node = _load_manifest_node(project_dir, model_name)
    inferred_index, inferred_schema = _resolve_index_schema_from_node(node)
    if set_index and inferred_index:
        env["INDEX_NAME"] = inferred_index
    if set_schema and inferred_schema:
        env["SCHEMA"] = inferred_schema
    return bool(inferred_index), bool(inferred_schema)


def _build_embed_env(
    cwd: Path,
    profiles_dir: str,
    profile_name: str | None,
    target_name: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    profile_file = env.get("PROFILE_FILE") or str(Path(profiles_dir) / "profiles.yml")
    chosen_profile = profile_name or env.get("PROFILE", "default")
    chosen_target = target_name or env.get("TARGET")

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
            profile = data.get(chosen_profile, {}) or {}
            outputs = profile.get("outputs", {}) or {}
            target = chosen_target or profile.get("target")
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
                if not env.get("PGDATABASE"):
                    db_name = cfg.get("dbname")
                    if db_name is None:
                        db_name = cfg.get("database")
                    if db_name is not None:
                        env["PGDATABASE"] = str(db_name)
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
    env.setdefault("EMBED_DB_BATCH_SIZE", "1000")
    return env


def _apply_embed_overrides(env: dict[str, str], args: argparse.Namespace) -> None:
    if getattr(args, "embed_db_batch_size", None):
        env["EMBED_DB_BATCH_SIZE"] = str(args.embed_db_batch_size)
    if getattr(args, "embed_max_batch", None):
        env["EMBED_MAX_BATCH"] = str(args.embed_max_batch)
    if getattr(args, "embed_dims", None):
        env["EMBED_DIMS"] = str(args.embed_dims)
    if getattr(args, "embed_provider", None):
        env["EMBED_PROVIDER"] = args.embed_provider
    if getattr(args, "embed_model", None):
        env["EMBED_MODEL"] = args.embed_model
    if getattr(args, "embed_local_model_path", None):
        env["EMBED_LOCAL_MODEL_PATH"] = args.embed_local_model_path
    if getattr(args, "index_name", None):
        env["INDEX_NAME"] = args.index_name
    if getattr(args, "schema", None):
        env["SCHEMA"] = args.schema
    if getattr(args, "limit", None) is not None:
        env["EMBED_LIMIT"] = str(args.limit)
    if getattr(args, "verbose", False):
        env["EMBED_LOG_BATCHES"] = "1"
    if getattr(args, "keep_source", False):
        env["EMBED_KEEP_SOURCE"] = "1"


def _handle_build(args: argparse.Namespace) -> int:
    cwd = _resolve_cwd()
    dbt_cmd = os.environ.get("DBT", "dbt")
    profiles_dir = str(Path(args.profiles_dir or os.environ.get("PROFILE_DIR") or os.environ.get("DBT_PROFILES_DIR") or cwd).expanduser())
    project_dir = str(Path(args.project_dir or os.environ.get("PROJECT_DIR") or cwd).expanduser())
    selector = args.select or os.environ.get("SELECT_MODEL", "vector_knowledge_base")

    env = _build_embed_env(cwd, profiles_dir, args.profile, args.target)
    _apply_embed_overrides(env, args)

    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be greater than 0")
    if args.limit and not args.allow_partial:
        raise RuntimeError(
            "--limit only applies to embedding writes and can leave partial placeholder rows. "
            "Re-run with --allow-partial to confirm."
        )

    project_path = Path(project_dir)
    if not _has_vector_index_materialization(project_path):
        raise RuntimeError(
            "vector_index materialization not found in project macros/dbt_packages. "
            "Run `dbt-vectorize init --project-dir ...` and then `dbt deps`."
        )

    resolved_model = _resolve_model_name(
        dbt_cmd=dbt_cmd,
        selector=selector,
        profiles_dir=profiles_dir,
        project_dir=project_dir,
        profile_name=args.profile,
        target_name=args.target,
        env=env,
        cwd=cwd,
    )

    # Best-effort pre-run inference (dry-run/helpful logging).
    _apply_manifest_target(
        project_path,
        resolved_model,
        env,
        set_index=not bool(args.index_name),
        set_schema=not bool(args.schema),
    )

    if args.dry_run:
        print(f"[build] selector={selector}", flush=True)
        print(f"[build] resolved_model={resolved_model}", flush=True)
        print(f"[build] index={env.get('INDEX_NAME')} schema={env.get('SCHEMA')}", flush=True)
        print(
            f"[build] provider={env.get('EMBED_PROVIDER')} model={env.get('EMBED_MODEL')} "
            f"dims={env.get('EMBED_DIMS')} db_batch={env.get('EMBED_DB_BATCH_SIZE')} "
            f"provider_batch={env.get('EMBED_MAX_BATCH', 'default')}",
            flush=True,
        )
        return 0

    run_cmd = [
        dbt_cmd,
        "run",
        "--no-partial-parse",
        "--select",
        selector,
        *_dbt_flags(profiles_dir, project_dir, args.profile, args.target),
    ]
    if args.vars:
        run_cmd.extend(["--vars", args.vars])
    if args.full_refresh:
        run_cmd.append("--full-refresh")

    print(
        f"[build] running dbt selector {selector} (provider={env.get('EMBED_PROVIDER')}, model={env.get('EMBED_MODEL')})",
        flush=True,
    )
    proc = subprocess.run(run_cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        return proc.returncode

    # Post-run inference is authoritative (manifest just refreshed by dbt).
    have_index, _ = _apply_manifest_target(
        project_path,
        resolved_model,
        env,
        set_index=not bool(args.index_name),
        set_schema=not bool(args.schema),
    )
    if not args.index_name and not have_index:
        raise RuntimeError(
            f"Could not infer index_name for model '{resolved_model}'. "
            "Pass --index-name explicitly."
        )

    embed_cmd, embed_cwd = _find_binary_cmd("pg_embedder", cwd)
    print(f"[build] generating embeddings into {env['SCHEMA']}.{env['INDEX_NAME']}", flush=True)
    proc = subprocess.run(embed_cmd, cwd=str(embed_cwd or cwd), env=env)
    if proc.returncode != 0:
        return proc.returncode
    print("[build] done.", flush=True)
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    cwd = _resolve_cwd()
    dbt_cmd = os.environ.get("DBT", "dbt")
    profiles_dir = str(Path(args.profiles_dir or os.environ.get("PROFILE_DIR") or os.environ.get("DBT_PROFILES_DIR") or cwd).expanduser())
    project_dir = str(Path(args.project_dir or os.environ.get("PROJECT_DIR") or cwd).expanduser())
    env = _build_embed_env(cwd, profiles_dir, args.profile, args.target)
    _apply_embed_overrides(env, args)
    if args.top_k <= 0:
        raise RuntimeError("--top-k must be greater than 0")

    resolved_model = None
    if args.select:
        resolved_model = _resolve_model_name(
            dbt_cmd=dbt_cmd,
            selector=args.select,
            profiles_dir=profiles_dir,
            project_dir=project_dir,
            profile_name=args.profile,
            target_name=args.target,
            env=env,
            cwd=cwd,
        )
        have_index, _ = _apply_manifest_target(
            Path(project_dir),
            resolved_model,
            env,
            set_index=not bool(args.index),
            set_schema=not bool(args.schema),
        )
        if not args.index and not have_index:
            raise RuntimeError(
                f"Could not infer index_name for model '{resolved_model}'. "
                "Pass --index explicitly."
            )
    elif not args.index:
        raise RuntimeError("search requires either --select or --index")

    if args.index:
        env["INDEX_NAME"] = args.index
    if args.schema:
        env["SCHEMA"] = args.schema
    if args.columns:
        cols = _validated_columns_arg(args.columns)
        if cols:
            env["VECTOR_SEARCH_COLUMNS"] = cols

    cmd, search_cwd = _find_binary_cmd("pg_search", cwd)
    cmd.extend(
        [
            "--schema",
            env["SCHEMA"],
            "--index",
            env["INDEX_NAME"],
            "--query",
            args.query,
            "--top-k",
            str(args.top_k),
            "--format",
            args.format,
        ]
    )
    if args.include_distance:
        cmd.append("--include-distance")

    if resolved_model:
        print(
            f"[search] resolved model {resolved_model} -> {env['SCHEMA']}.{env['INDEX_NAME']}",
            flush=True,
        )
    else:
        print(f"[search] using index {env['SCHEMA']}.{env['INDEX_NAME']}", flush=True)
    proc = subprocess.run(cmd, cwd=str(search_cwd or cwd), env=env)
    return proc.returncode


def _handle_embed(args: argparse.Namespace) -> int:
    cwd = _resolve_cwd()
    dbt_cmd = os.environ.get("DBT", "dbt")
    profiles_dir = str(
        Path(
            args.profiles_dir
            or os.environ.get("PROFILE_DIR")
            or os.environ.get("DBT_PROFILES_DIR")
            or cwd
        ).expanduser()
    )
    project_dir = str(
        Path(args.project_dir or os.environ.get("PROJECT_DIR") or cwd).expanduser()
    )
    env = _build_embed_env(cwd, profiles_dir, args.profile, args.target)
    _apply_embed_overrides(env, args)
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be greater than 0")
    if args.limit and not args.allow_partial:
        raise RuntimeError(
            "--limit only applies to embedding writes and can leave partial placeholder rows. "
            "Re-run with --allow-partial to confirm."
        )

    if args.select:
        resolved_model = _resolve_model_name(
            dbt_cmd=dbt_cmd,
            selector=args.select,
            profiles_dir=profiles_dir,
            project_dir=project_dir,
            profile_name=args.profile,
            target_name=args.target,
            env=env,
            cwd=cwd,
        )
        have_index, _ = _apply_manifest_target(
            Path(project_dir),
            resolved_model,
            env,
            set_index=not bool(args.index_name),
            set_schema=not bool(args.schema),
        )
        if not args.index_name and not have_index:
            raise RuntimeError(
                f"Could not infer index_name for model '{resolved_model}'. "
                "Pass --index-name explicitly."
            )
        print(
            f"[embed] resolved model {resolved_model} -> {env['SCHEMA']}.{env['INDEX_NAME']}",
            flush=True,
        )
    elif not args.index_name:
        raise RuntimeError("embed requires either --select or --index-name")

    if args.index_name:
        env["INDEX_NAME"] = args.index_name
    if args.schema:
        env["SCHEMA"] = args.schema

    embed_cmd, embed_cwd = _find_binary_cmd("pg_embedder", cwd)
    print(f"[embed] generating embeddings into {env['SCHEMA']}.{env['INDEX_NAME']}", flush=True)
    proc = subprocess.run(embed_cmd, cwd=str(embed_cwd or cwd), env=env)
    if proc.returncode != 0:
        return proc.returncode
    print("[embed] done.", flush=True)
    return 0


def _handle_inspect(args: argparse.Namespace) -> int:
    cwd = _resolve_cwd()
    dbt_cmd = os.environ.get("DBT", "dbt")
    profiles_dir = str(Path(args.profiles_dir or os.environ.get("PROFILE_DIR") or os.environ.get("DBT_PROFILES_DIR") or cwd).expanduser())
    project_dir = str(Path(args.project_dir or os.environ.get("PROJECT_DIR") or cwd).expanduser())
    env = _build_embed_env(cwd, profiles_dir, args.profile, args.target)

    model = _resolve_model_name(
        dbt_cmd=dbt_cmd,
        selector=args.select,
        profiles_dir=profiles_dir,
        project_dir=project_dir,
        profile_name=args.profile,
        target_name=args.target,
        env=env,
        cwd=cwd,
    )
    node = _load_manifest_node(Path(project_dir), model)
    if not node:
        raise RuntimeError("Could not load target/manifest.json model node.")
    cfg = node.get("config") or {}
    index_name = cfg.get("index_name") or node.get("alias") or node.get("name")
    print(f"model: {model}")
    print(f"schema: {node.get('schema')}")
    print(f"index_name: {index_name}")
    print(f"materialized: {cfg.get('materialized')}")
    print(f"unique_key: {cfg.get('unique_key')}")
    print(f"text_column: {cfg.get('text_column')}")
    print(f"dimensions: {cfg.get('dimensions')}")
    print(f"embedding_model: {cfg.get('embedding_model')}")
    print(f"metadata_columns: {cfg.get('metadata_columns')}")
    print(f"updated_at_column: {cfg.get('updated_at_column')}")
    return 0


def _handle_init(args: argparse.Namespace) -> int:
    cwd = _resolve_cwd()
    dbt_cmd = os.environ.get("DBT", "dbt")
    project_dir = Path(args.project_dir or os.environ.get("PROJECT_DIR") or cwd).expanduser()
    project_dir = project_dir.resolve()
    if not (project_dir / "dbt_project.yml").exists():
        raise RuntimeError(f"dbt_project.yml not found in {project_dir}")

    git_url = args.git_url
    revision = args.revision
    packages_file, changed = _ensure_package_entry(project_dir, git_url, revision)

    if changed:
        print(f"[init] updated {packages_file}", flush=True)
    else:
        print(f"[init] packages.yml already configured: {packages_file}", flush=True)

    if args.skip_deps:
        print("[init] skipping dbt deps (--skip-deps)", flush=True)
        return 0

    cmd = [dbt_cmd, "deps", "--project-dir", str(project_dir)]
    print(f"[init] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(project_dir), env=os.environ.copy())
    if proc.returncode != 0:
        return proc.returncode
    print("[init] done.", flush=True)
    return 0


def _add_common_build_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--select", default=os.environ.get("SELECT_MODEL", "vector_knowledge_base"))
    parser.add_argument("--profiles-dir")
    parser.add_argument("--project-dir")
    parser.add_argument("--profile")
    parser.add_argument("--target")
    parser.add_argument("--vars")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument("--embed-db-batch-size", type=int)
    parser.add_argument("--embed-max-batch", type=int)
    parser.add_argument("--embed-dims", type=int)
    parser.add_argument("--embed-provider")
    parser.add_argument("--embed-model")
    parser.add_argument("--embed-local-model-path")
    parser.add_argument("--index-name")
    parser.add_argument("--schema")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbt-vectorize",
        description="Build and query pgvector indexes from dbt models.",
    )
    parser.add_argument("--version", action="version", version=f"dbt-vectorize {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Run dbt model and generate embeddings")
    _add_common_build_flags(p_build)

    p_search = sub.add_parser("search", help="Semantic search against a vector index")
    p_search.add_argument("--select")
    p_search.add_argument("--index")
    p_search.add_argument("--schema")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--format", choices=["table", "json"], default="table")
    p_search.add_argument("--include-distance", action="store_true")
    p_search.add_argument("--columns", help="Comma-separated result columns to include (e.g. doc_id,text,source,created_at)")
    p_search.add_argument("--profiles-dir")
    p_search.add_argument("--project-dir")
    p_search.add_argument("--profile")
    p_search.add_argument("--target")
    p_search.add_argument("--embed-provider")
    p_search.add_argument("--embed-model")
    p_search.add_argument("--embed-local-model-path")
    p_search.add_argument("--embed-dims", type=int)
    p_search.add_argument("--embed-max-batch", type=int)

    p_embed = sub.add_parser("embed", help="Run embedding/upsert only (skip dbt run)")
    p_embed.add_argument("--select")
    p_embed.add_argument("--index-name")
    p_embed.add_argument("--schema")
    p_embed.add_argument("--profiles-dir")
    p_embed.add_argument("--project-dir")
    p_embed.add_argument("--profile")
    p_embed.add_argument("--target")
    p_embed.add_argument("--limit", type=int)
    p_embed.add_argument("--allow-partial", action="store_true")
    p_embed.add_argument("--verbose", action="store_true")
    p_embed.add_argument("--keep-source", action="store_true")
    p_embed.add_argument("--embed-db-batch-size", type=int)
    p_embed.add_argument("--embed-max-batch", type=int)
    p_embed.add_argument("--embed-dims", type=int)
    p_embed.add_argument("--embed-provider")
    p_embed.add_argument("--embed-model")
    p_embed.add_argument("--embed-local-model-path")

    p_inspect = sub.add_parser("inspect", help="Show resolved model vector config")
    p_inspect.add_argument("--select", required=True)
    p_inspect.add_argument("--profiles-dir")
    p_inspect.add_argument("--project-dir")
    p_inspect.add_argument("--profile")
    p_inspect.add_argument("--target")

    p_init = sub.add_parser("init", help="Configure packages.yml and run dbt deps")
    p_init.add_argument("--project-dir")
    p_init.add_argument("--git-url", default="https://github.com/kraftaa/dbt-vector.git")
    p_init.add_argument("--revision", default=f"v{__version__}")
    p_init.add_argument("--skip-deps", action="store_true")

    return parser


def main() -> int:
    raw = sys.argv[1:]
    if not raw:
        raw = ["build"]
    elif raw[0] not in SUBCOMMANDS and raw[0] not in {"-h", "--help", "--version"}:
        raw = ["build", *raw]

    parser = _build_parser()
    args = parser.parse_args(raw)

    try:
        if args.command == "build":
            return _handle_build(args)
        if args.command == "embed":
            return _handle_embed(args)
        if args.command == "search":
            return _handle_search(args)
        if args.command == "inspect":
            return _handle_inspect(args)
        if args.command == "init":
            return _handle_init(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
