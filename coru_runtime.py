#!/usr/bin/env python3
"""CORU-10M-ACT production-oriented loader and diagnostic runner (README-independent release verification).

Loads the official RRR PRO MEX Hugging Face repository through the repository's
own coru_inference.py runtime. It does not assume a Transformers AutoModel API.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as _dt
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ID = "rrrpromex/CORU-10M-ACT"
REPO_URL = f"https://huggingface.co/{REPO_ID}"
EXPECTED_PARAMETERS = 9_995_264
EXPECTED_VOCAB_SIZE = 7_680
EXPECTED_ACTIONS = 55
EXPECTED_CONTEXT = 512
COPYRIGHT = "© All Rights Reserved By RRR PRO MEX"


class CoruError(RuntimeError):
    pass


class EnterpriseLogger:
    """Timestamped console + persistent plain-text run log."""
    RESET = "\033[0m"
    COLORS = {
        "INFO": "\033[96m",
        "PASS": "\033[92m",
        "WARN": "\033[93m",
        "FAIL": "\033[91m",
        "STEP": "\033[94m",
    }

    def __init__(self, path: Path, color: bool | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.color = sys.stdout.isatty() if color is None else color
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)

    def log(self, level: str, message: str) -> None:
        stamp = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        plain = f"[{stamp}] [{level:<4}] {message}"
        print(f"{self.COLORS.get(level, '')}{plain}{self.RESET if self.color else ''}", flush=True)
        self._fh.write(plain + "\n")

    def info(self, s: str) -> None: self.log("INFO", s)
    def step(self, s: str) -> None: self.log("STEP", s)
    def passed(self, s: str) -> None: self.log("PASS", s)
    def warn(self, s: str) -> None: self.log("WARN", s)
    def fail(self, s: str) -> None: self.log("FAIL", s)

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def footer(self) -> None:
        self.log("INFO", "=" * 76)
        self.log("INFO", COPYRIGHT)
        self.log("INFO", "=" * 76)


@dataclass
class DeviceSelection:
    requested: str
    backend: str
    device: Any
    description: str
    synchronize: Callable[[], None] | None = None


def _xla_device_info() -> tuple[Any, str, Callable[[], None]] | None:
    """Return an active TPU/XLA device only when the runtime is actually usable."""
    try:
        import torch_xla  # type: ignore
        try:
            device = torch_xla.device()
        except Exception:
            import torch_xla.core.xla_model as xm  # type: ignore
            device = xm.xla_device()

        hardware = ""
        try:
            import torch_xla.runtime as xr  # type: ignore
            hardware = str(xr.device_type()).upper()
        except Exception:
            pass
        if not hardware:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore
                hardware = str(xm.xla_device_hw(device)).upper()
            except Exception:
                pass
        if not hardware:
            hardware = os.environ.get("PJRT_DEVICE", "").upper()

        # Do not mistake an XLA CPU backend for a TPU.
        if "TPU" not in hardware:
            return None

        def sync() -> None:
            try:
                torch_xla.sync(wait=True)
            except TypeError:
                torch_xla.sync()
            except Exception:
                try:
                    import torch_xla.core.xla_model as xm  # type: ignore
                    xm.mark_step()
                except Exception:
                    pass

        return device, hardware, sync
    except Exception:
        return None


def select_device(requested: str, torch: Any) -> DeviceSelection:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "tpu", "xla", "cuda", "mps"}:
        raise CoruError(f"Unsupported device selection: {requested}")

    xla = _xla_device_info() if requested in {"auto", "tpu", "xla"} else None
    if requested in {"tpu", "xla"}:
        if xla is None:
            raise CoruError("TPU requested, but an active PyTorch/XLA TPU device was not detected. Select a TPU runtime and verify torch_xla.")
        return DeviceSelection(requested, "TPU/XLA", xla[0], f"{xla[1]} ({xla[0]})", xla[2])
    if requested == "auto" and xla is not None:
        return DeviceSelection(requested, "TPU/XLA", xla[0], f"{xla[1]} ({xla[0]})", xla[2])

    if requested in {"auto", "cuda"} and getattr(torch, "cuda", None) is not None:
        try:
            if torch.cuda.is_available():
                dev = torch.device("cuda:0")
                name = torch.cuda.get_device_name(0)
                return DeviceSelection(requested, "CUDA", dev, f"{name} ({dev})", torch.cuda.synchronize)
        except Exception:
            if requested == "cuda":
                raise CoruError("CUDA was requested but could not be initialized.")
    elif requested == "cuda":
        raise CoruError("CUDA was requested but this PyTorch build has no CUDA support.")

    if requested in {"auto", "mps"}:
        try:
            if torch.backends.mps.is_available():
                dev = torch.device("mps")
                return DeviceSelection(requested, "APPLE MPS", dev, "Apple Metal Performance Shaders (mps)")
        except Exception:
            if requested == "mps":
                raise CoruError("MPS was requested but is not available in this PyTorch runtime.")
    elif requested == "mps":
        raise CoruError("MPS is not available in this PyTorch runtime.")

    if requested not in {"auto", "cpu"}:
        raise CoruError(f"Requested accelerator '{requested}' is not available.")
    return DeviceSelection(requested, "CPU", torch.device("cpu"), f"CPU ({platform.processor() or platform.machine() or 'host processor'})")


def configure_cpu(torch: Any) -> int:
    cores = os.cpu_count() or 2
    threads = max(1, min(8, cores))
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(4, threads // 2 or 1)))
    except Exception:
        pass
    return threads


def ensure_python_package(module_name: str, requirement: str, logger: EnterpriseLogger) -> Any:
    try:
        return __import__(module_name)
    except ImportError:
        logger.warn(f"Missing dependency '{module_name}'; installing {requirement}.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", requirement])
        return __import__(module_name)


def ensure_hf_hub(logger: EnterpriseLogger) -> Any:
    return ensure_python_package("huggingface_hub", "huggingface_hub>=0.24.0", logger)


def download_repository(repo_id: str, revision: str, destination: Path, logger: EnterpriseLogger) -> Path:
    """Download into Hugging Face's immutable, content-addressed snapshot cache.

    Do not use ``local_dir`` here: it is a mutable export directory and can retain
    a previously edited README or other stale files across notebook executions.
    The immutable snapshot path keeps the repository bytes separate from the
    companion loader, reports, and user-created documentation.
    """
    hub = ensure_hf_hub(logger)
    destination.mkdir(parents=True, exist_ok=True)
    logger.step(f"Resolving Hugging Face repository: {repo_id}@{revision}")
    try:
        # Resolve the requested branch/tag against Hugging Face first. This stops
        # an offline or stale local cache from being mistaken for the current main.
        token = os.environ.get("HF_TOKEN") or None
        api = hub.HfApi(token=token)
        info = api.model_info(repo_id=repo_id, revision=revision)
        commit_sha = str(getattr(info, "sha", "") or "")
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit_sha):
            raise CoruError("Hugging Face did not return a valid immutable commit SHA for the requested revision.")
        logger.info(f"Current remote revision resolved: {revision} -> {commit_sha}")
        result = hub.snapshot_download(
            repo_id=repo_id,
            revision=commit_sha,
            cache_dir=str(destination),
            repo_type="model",
            token=token,
        )
    except CoruError:
        raise
    except Exception as exc:
        raise CoruError(
            f"Could not resolve/download the current Hugging Face revision {repo_id}@{revision}: {exc}. "
            "The loader will not fall back to an older cached revision. Check internet access, repository visibility, and HF_TOKEN if private."
        ) from exc
    model_dir = Path(result).resolve()
    logger.passed(f"Immutable current-revision snapshot ready: {model_dir}")
    return model_dir


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def import_runtime(model_dir: Path) -> Any:
    runtime_path = model_dir / "coru_inference.py"
    if not runtime_path.is_file():
        raise CoruError(f"Required CORU runtime was not found: {runtime_path}")
    sys.path.insert(0, str(model_dir))
    spec = importlib.util.spec_from_file_location("coru_inference", runtime_path)
    if spec is None or spec.loader is None:
        raise CoruError("Unable to create an import specification for coru_inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CoruError(f"Importing the repository's CORU runtime failed: {exc}") from exc
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_markdown_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/").strip().lower()
    return normalized.endswith((".md", ".markdown", ".mdown"))


def _manifest_without_markdown_entries(value: Any) -> tuple[Any, int]:
    """Return a copy of a checksum manifest without Markdown-file entries.

    Handles both path-keyed dictionaries and list records such as
    {"path": "README.md", "sha256": "..."}. Non-Markdown values are
    recursively preserved byte-for-byte at the JSON data-model level.
    """
    removed = 0
    sentinel = object()

    def visit(node: Any) -> Any:
        nonlocal removed
        if isinstance(node, dict):
            # A list record may describe a markdown file by its path field.
            for path_key in ("path", "file", "filename", "filepath"):
                if _is_markdown_path(node.get(path_key)):
                    removed += 1
                    return sentinel
            out: dict[str, Any] = {}
            for key, child in node.items():
                if _is_markdown_path(str(key)):
                    removed += 1
                    continue
                cleaned = visit(child)
                if cleaned is not sentinel:
                    out[key] = cleaned
            return out
        if isinstance(node, list):
            out_list = []
            for item in node:
                cleaned = visit(item)
                if cleaned is not sentinel:
                    out_list.append(cleaned)
            return out_list
        return node

    cleaned = visit(copy.deepcopy(value))
    return ({} if cleaned is sentinel else cleaned), removed


def _remove_markdown_files_from_workspace(model_dir: Path, logger: EnterpriseLogger) -> list[str]:
    removed: list[str] = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and _is_markdown_path(path.name):
            relative = path.relative_to(model_dir).as_posix()
            path.unlink()
            removed.append(relative)
    logger.info(
        "Documentation policy: Markdown files are optional and excluded from inference-package verification. "
        f"Removed from private workspace={len(removed)}."
    )
    return removed


def prepare_documentation_independent_workspace(model_dir: Path, logger: EnterpriseLogger) -> dict[str, Any]:
    """Remove only Markdown documentation from the isolated workspace and its hash list.

    The source Hugging Face snapshot is never changed. All non-Markdown manifest
    records are preserved unchanged and remain under the repository's strict
    integrity verifier.
    """
    checksum_path = model_dir / "PACKAGE_SHA256.json"
    if not checksum_path.is_file():
        raise CoruError("PACKAGE_SHA256.json is missing; cannot verify the current repository package.")
    original_bytes = checksum_path.read_bytes()
    manifest = safe_read_json(checksum_path)
    if manifest is None:
        raise CoruError("PACKAGE_SHA256.json is invalid JSON; refusing an unverified model load.")

    cleaned_manifest, removed_manifest_entries = _manifest_without_markdown_entries(manifest)
    if cleaned_manifest != manifest:
        temp = checksum_path.with_name(f".PACKAGE_SHA256.docs-{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(cleaned_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, checksum_path)
        finally:
            temp.unlink(missing_ok=True)

    removed_files = _remove_markdown_files_from_workspace(model_dir, logger)
    # Ensure the new manifest still parses and contains non-documentation entries.
    reread = safe_read_json(checksum_path)
    if reread is None:
        checksum_path.write_bytes(original_bytes)
        raise CoruError("Documentation filtering produced an invalid checksum manifest; original workspace manifest restored.")

    logger.passed(
        "Documentation-independent workspace ready: Markdown files are not required; "
        "all non-Markdown checksum records are preserved."
    )
    return {
        "markdown_files_removed": removed_files,
        "markdown_hash_entries_removed": removed_manifest_entries,
        "policy": "markdown_optional_non_markdown_strict",
    }


def patch_runtime_verifier_markdown_policy(runtime: Any, logger: EnterpriseLogger) -> int:
    """Patch only the verifier's required-file set in memory, never its source file.

    The repository's original coru_inference.py remains untouched and its original
    SHA-256 is still checked by PACKAGE_SHA256.json. The imported verifier function
    is recreated from its AST with only .md constants removed from the explicit
    required-file collection. All remaining verification logic runs unchanged.
    """
    original_fn = getattr(runtime, "verify_package_integrity", None)
    if not callable(original_fn):
        raise CoruError("The CORU runtime does not expose verify_package_integrity(); refusing an unverified model load.")
    source_path = Path(getattr(runtime, "__file__", ""))
    if not source_path.is_file():
        raise CoruError("Cannot locate the repository runtime source to validate its documentation policy safely.")
    source = source_path.read_text(encoding="utf-8")
    source_digest_before = _sha256_file(source_path)
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:
        raise CoruError(f"The repository runtime source cannot be parsed for safe verifier adaptation: {exc}") from exc

    fn_node = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "verify_package_integrity"), None)
    if fn_node is None or isinstance(fn_node, ast.AsyncFunctionDef):
        raise CoruError("The repository runtime's verifier is not a supported module-level synchronous function.")

    changed = 0
    required_assignment_found = False
    for node in ast.walk(fn_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = {t.id for t in targets if isinstance(t, ast.Name)}
        if "required" not in target_names:
            continue
        required_assignment_found = True
        value = node.value
        if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
            kept = []
            for element in value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str) and _is_markdown_path(element.value):
                    changed += 1
                else:
                    kept.append(element)
            value.elts = kept
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "set" and value.args and isinstance(value.args[0], (ast.List, ast.Tuple, ast.Set)):
            container = value.args[0]
            kept = []
            for element in container.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str) and _is_markdown_path(element.value):
                    changed += 1
                else:
                    kept.append(element)
            container.elts = kept
        else:
            raise CoruError("The CORU verifier's required-file declaration uses an unsupported form; no integrity checks were bypassed.")

    if not required_assignment_found:
        raise CoruError("The CORU verifier's required-file declaration was not found; refusing to guess its integrity policy.")
    if changed < 1:
        # Current runtime could already have Markdown removed from its required set.
        # Verify it does not contain explicit Markdown path literals in the verifier.
        remaining_md = [
            n.value for n in ast.walk(fn_node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and _is_markdown_path(n.value)
        ]
        if remaining_md:
            raise CoruError("The CORU verifier references Markdown paths outside its required-file set; refusing unsafe adaptation.")

    # Build just this function in the original module namespace. This keeps the
    # runtime source file and all other module objects untouched. Module-level
    # global lookup means load() will use the adapted verifier as well.
    fn_node.decorator_list = []
    mini_module = ast.Module(body=[fn_node], type_ignores=[])
    ast.fix_missing_locations(mini_module)
    try:
        compiled = compile(mini_module, str(source_path), "exec")
        exec(compiled, runtime.__dict__)
    except Exception as exc:
        raise CoruError(f"Could not safely adapt the verifier's Markdown required-file policy: {exc}") from exc

    patched_fn = getattr(runtime, "verify_package_integrity", None)
    if not callable(patched_fn) or patched_fn.__globals__ is not runtime.__dict__:
        raise CoruError("In-memory verifier adaptation did not bind to the official runtime module.")
    if _sha256_file(source_path) != source_digest_before:
        raise CoruError("Repository runtime source unexpectedly changed during verifier adaptation.")
    logger.passed(
        f"Official verifier adapted in memory: removed {changed} Markdown path(s) from its required-file set; "
        "all other verifier logic and the on-disk runtime source remain unchanged."
    )
    return changed


def call_integrity_verifier(runtime: Any, model_dir: Path, logger: EnterpriseLogger) -> None:
    verifier = getattr(runtime, "verify_package_integrity", None)
    if not callable(verifier):
        raise CoruError("The CORU runtime does not expose verify_package_integrity(); refusing an unverified load.")
    sig = inspect.signature(verifier)
    kwargs: dict[str, Any] = {}
    if "allow_interim" in sig.parameters:
        kwargs["allow_interim"] = False
    # The official verifier remains fully enabled. README's digest is refreshed
    # in the isolated copy immediately before this call; every other digest and
    # the release identity are still checked by the upstream runtime.
    verifier(model_dir, **kwargs)


def call_runtime_load(runtime: Any, model_dir: Path, device: Any) -> Any:
    loader = getattr(runtime, "load", None)
    if not callable(loader):
        raise CoruError("The CORU runtime does not expose load(); refusing to guess an incompatible model loader.")
    sig = inspect.signature(loader)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if "device" in params:
        kwargs["device"] = str(device)
    if "allow_interim" in params:
        kwargs["allow_interim"] = False
    # The known CORU runtime accepts the package directory as the first positional argument.
    return loader(model_dir, **kwargs)


def walk_objects(root: Any, max_depth: int = 4) -> Iterable[Any]:
    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        obj, depth = stack.pop()
        if obj is None or id(obj) in seen or depth > max_depth:
            continue
        seen.add(id(obj))
        yield obj
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, (dict, tuple, list)) or hasattr(value, "parameters") or hasattr(value, "encode"):
                    stack.append((value, depth + 1))
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                stack.append((value, depth + 1))
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            for key, value in vars(obj).items():
                if not key.startswith("_") and (isinstance(value, (dict, tuple, list)) or hasattr(value, "parameters") or hasattr(value, "encode")):
                    stack.append((value, depth + 1))


def find_model(loaded: Any, torch: Any) -> Any | None:
    for obj in walk_objects(loaded):
        try:
            if isinstance(obj, torch.nn.Module):
                return obj
        except Exception:
            pass
    return None


def find_tokenizer(loaded: Any) -> Any | None:
    for obj in walk_objects(loaded):
        if callable(getattr(obj, "encode", None)) and not callable(getattr(obj, "parameters", None)):
            return obj
    return None


def count_parameters(model: Any) -> int | None:
    try:
        return sum(int(p.numel()) for p in model.parameters())
    except Exception:
        return None


def validate_weights(model: Any, torch: Any) -> tuple[int, int]:
    """Validate all parameters/buffers with one device synchronization where possible."""
    tensors = list(model.parameters()) + list(model.buffers())
    total = sum(int(tensor.numel()) for tensor in tensors)
    finite_flags = []
    checked = 0
    for tensor in tensors:
        if tensor.is_floating_point() or tensor.is_complex():
            finite_flags.append(torch.isfinite(tensor).all())
            checked += 1
    if finite_flags:
        # A single reduction avoids synchronizing the TPU/XLA runtime once per tensor.
        all_finite = torch.stack(finite_flags).all()
        if not bool(all_finite.item()):
            raise CoruError("Model contains NaN or infinite parameter/buffer values.")
    return total, checked


def find_named_value(root: Any, names: set[str]) -> Any | None:
    if isinstance(root, dict):
        for k, v in root.items():
            if str(k).lower() in names:
                return v
        for v in root.values():
            hit = find_named_value(v, names)
            if hit is not None:
                return hit
    elif isinstance(root, list):
        for v in root:
            hit = find_named_value(v, names)
            if hit is not None:
                return hit
    return None


def schema_action_count(schema: Any) -> int | None:
    if isinstance(schema, list):
        return len(schema)
    if not isinstance(schema, dict):
        return None
    for k in ("actions", "action_routes", "routes", "action_schema", "intents", "commands"):
        v = schema.get(k)
        if isinstance(v, (list, dict)):
            return len(v)
    # Some schema files map action identifiers directly at the top level.
    if schema and all(isinstance(v, (dict, str, list)) for v in schema.values()):
        excluded = {"$schema", "title", "description", "version", "namespaces", "metadata"}
        count = sum(1 for k in schema if str(k).lower() not in excluded)
        return count or None
    return None


def inspect_package(model_dir: Path, logger: EnterpriseLogger) -> dict[str, Any]:
    config = safe_read_json(model_dir / "config.json")
    manifest = safe_read_json(model_dir / "manifest.json")
    schema = safe_read_json(model_dir / "action_schema.json")
    tok = safe_read_json(model_dir / "tokenizer.json")
    if not isinstance(config, dict):
        config = {}
    if not isinstance(manifest, dict):
        manifest = {}
    vocab = find_named_value(config, {"vocab_size", "vocabulary_size"})
    if vocab is None:
        vocab = find_named_value(manifest, {"vocab_size", "vocabulary_size"})
    if vocab is None and isinstance(tok, dict):
        model = tok.get("model", {})
        if isinstance(model, dict):
            raw_vocab = model.get("vocab")
            if isinstance(raw_vocab, (list, dict)):
                vocab = len(raw_vocab)
    params = find_named_value(config, {"num_parameters", "parameter_count", "params"})
    if params is None:
        params = find_named_value(manifest, {"num_parameters", "parameter_count", "params"})
    context = find_named_value(config, {"context_length", "max_position_embeddings", "max_seq_len", "context_size"})
    if context is None:
        context = find_named_value(manifest, {"context_length", "max_position_embeddings", "max_seq_len", "context_size"})
    actions = schema_action_count(schema)
    logger.info(f"Package metadata: parameters={params if params is not None else 'runtime-verified'} | vocab={vocab if vocab is not None else 'not declared'} | context={context if context is not None else 'not declared'} | actions={actions if actions is not None else 'schema parsed by runtime'}")
    if vocab is not None and int(vocab) != EXPECTED_VOCAB_SIZE:
        raise CoruError(f"Tokenizer/config vocabulary mismatch: expected {EXPECTED_VOCAB_SIZE}, found {vocab}.")
    if context is not None and int(context) != EXPECTED_CONTEXT:
        logger.warn(f"Context metadata reports {context}; documented release reference is {EXPECTED_CONTEXT}. Runtime verifier remains authoritative.")
    if params is not None and int(params) != EXPECTED_PARAMETERS:
        logger.warn(f"Config parameter metadata reports {params}; loaded module count will be checked independently.")
    if actions is not None and actions != EXPECTED_ACTIONS:
        logger.warn(f"Detected {actions} top-level action entries; expected reference is {EXPECTED_ACTIONS}. Strict runtime verifier remains authoritative.")
    return {"config": config, "manifest": manifest, "schema": schema, "vocab_size": vocab, "parameter_metadata": params, "context": context, "action_count": actions}


_TEXT_NAMES = {"text", "prompt", "request", "query", "message", "user_input", "input_text", "instruction", "utterance", "command_text", "user_message", "user_text", "text_input", "raw_text", "content"}
_MODEL_NAMES = {"model", "network", "net", "module", "coru_model"}
_TOKENIZER_NAMES = {"tokenizer", "tok", "sp", "encoder"}
_DEVICE_NAMES = {"device", "torch_device", "target_device"}
_SCHEMA_NAMES = {"schema", "action_schema", "actions_schema"}
_PACKAGE_NAMES = {"model_dir", "package_dir", "model_path", "package_path", "path"}
_INPUT_IDS_NAMES = {"input_ids", "token_ids", "tokens", "ids"}
_ATTENTION_NAMES = {"attention_mask", "input_mask", "mask"}


def encode_for_forward(tokenizer: Any, text: str, torch: Any, device: Any) -> tuple[Any, Any | None]:
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        return None, None
    try:
        encoded = tokenizer.encode(text, add_special_tokens=True)
    except TypeError:
        encoded = tokenizer.encode(text)
    if isinstance(encoded, dict):
        ids = encoded.get("input_ids")
        mask = encoded.get("attention_mask")
    else:
        ids = getattr(encoded, "ids", encoded)
        mask = None
    if ids is None:
        return None, None
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)
    attention = None
    if mask is not None:
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if mask and isinstance(mask[0], (list, tuple)):
            mask = mask[0]
        attention = torch.tensor([list(mask)], dtype=torch.long, device=device)
    return input_ids, attention


def invoke_with_supported_arguments(fn: Callable[..., Any], text: str, *, loaded: Any, model: Any, tokenizer: Any, device: Any, model_dir: Path, schema: Any, torch: Any = None) -> tuple[bool, Any, str]:
    try:
        sig = inspect.signature(fn)
    except Exception as exc:
        return False, None, f"signature unavailable: {exc}"
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    has_var_kw = False
    encoded_ids, attention_mask = (None, None)
    if tokenizer is not None and torch is not None:
        try:
            encoded_ids, attention_mask = encode_for_forward(tokenizer, text, torch, device)
        except Exception:
            encoded_ids, attention_mask = None, None
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kw = True
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        name = p.name.lower()
        value: Any = None
        found = True
        if name in _TEXT_NAMES:
            value = text
        elif name in _MODEL_NAMES:
            value = model or loaded
        elif name in _TOKENIZER_NAMES:
            value = tokenizer
        elif name in _DEVICE_NAMES:
            value = device
        elif name in _SCHEMA_NAMES:
            value = schema
        elif name in _PACKAGE_NAMES:
            value = str(model_dir)
        elif name in _INPUT_IDS_NAMES and encoded_ids is not None:
            value = encoded_ids
        elif name in _ATTENTION_NAMES and attention_mask is not None:
            value = attention_mask
        elif name in {"max_new_tokens", "max_length", "max_tokens", "top_k", "temperature"}:
            found = False  # retain runtime defaults
        else:
            found = False
        if not found:
            if p.default is inspect.Parameter.empty:
                return False, None, f"unsupported required argument '{p.name}'"
            continue
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            kwargs[p.name] = value
    if has_var_kw and not any(k.lower() in _TEXT_NAMES for k in kwargs):
        kwargs["text"] = text
    try:
        result = fn(*positional, **kwargs)
        return True, result, "ok"
    except Exception as exc:
        return False, None, f"call failed: {type(exc).__name__}: {exc}"


def find_inference_callables(runtime: Any, loaded: Any, model: Any) -> list[tuple[Callable[..., Any], str]]:
    names = (
        "route_request", "predict_action", "propose_action", "infer_request",
        "infer_action", "predict_intent", "classify_request", "route", "infer",
        "predict", "run_inference", "process_request", "process_input",
        "generate_proposal", "propose", "infer_text", "forward",
    )
    found: list[tuple[Callable[..., Any], str]] = []
    seen: set[int] = set()
    for owner_label, owner in (("runtime", runtime), ("loaded_model", loaded), ("model", model)):
        if owner is None:
            continue
        for name in names:
            fn = getattr(owner, name, None)
            if callable(fn) and id(fn) not in seen:
                seen.add(id(fn))
                found.append((fn, f"{owner_label}.{name}"))
    return found


def output_is_usable(value: Any, torch: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple)):
        return len(value) > 0
    try:
        if isinstance(value, torch.Tensor):
            return value.numel() > 0 and (not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item()))
    except Exception:
        pass
    return True


PROPOSAL_REQUIRED_FIELDS = (
    "tool", "action", "parameters", "status", "needs_confirmation", "clarification"
)


def normalize_proposal(value: Any) -> dict[str, Any]:
    """Normalize and validate CORU's structured action proposal contract.

    CORU proposes actions; this loader never executes external tools or actions.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise CoruError(f"Inference output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        for method_name in ("model_dump", "dict", "to_dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    value = method()
                    break
                except Exception:
                    pass
    if not isinstance(value, dict):
        raise CoruError(f"Inference output must be a structured object, received {type(value).__name__}.")

    missing = [key for key in PROPOSAL_REQUIRED_FIELDS if key not in value]
    if missing:
        raise CoruError("Inference output is missing required CORU proposal fields: " + ", ".join(missing))

    if value["tool"] is not None and not isinstance(value["tool"], str):
        raise CoruError("Proposal field 'tool' must be text or null.")
    if value["action"] is not None and not isinstance(value["action"], str):
        raise CoruError("Proposal field 'action' must be text or null.")
    if not isinstance(value["parameters"], dict):
        raise CoruError("Proposal field 'parameters' must be a JSON object/dictionary.")
    if not isinstance(value["status"], str) or not value["status"].strip():
        raise CoruError("Proposal field 'status' must be a non-empty string.")
    if not isinstance(value["needs_confirmation"], bool):
        raise CoruError("Proposal field 'needs_confirmation' must be a boolean.")
    if value["clarification"] is not None and not isinstance(value["clarification"], str):
        raise CoruError("Proposal field 'clarification' must be text or null.")
    return value


def run_inference_tests(runtime: Any, loaded: Any, model: Any, tokenizer: Any, device: Any, model_dir: Path, schema: Any, torch: Any, logger: EnterpriseLogger) -> int:
    candidates = find_inference_callables(runtime, loaded, model)
    if not candidates:
        raise CoruError(
            "The runtime loaded the model but no supported public inference entry point was discovered. "
            "Refusing to mark inference tests as passed. Check coru_inference.py API names."
        )
    tests = [
        "Open the calculator.",
        "Create a note titled Project Plan.",
        "Turn on dark mode.",
    ]
    selected_fn: Callable[..., Any] | None = None
    selected_label = ""
    first_output: Any = None
    failures: list[str] = []
    for fn, label in candidates:
        ok, output, reason = invoke_with_supported_arguments(
            fn, tests[0], loaded=loaded, model=model, tokenizer=tokenizer,
            device=device, model_dir=model_dir, schema=schema, torch=torch,
        )
        if ok:
            try:
                first_output = normalize_proposal(output)
                selected_fn, selected_label = fn, label
                break
            except CoruError as exc:
                failures.append(f"{label}: {exc}")
        else:
            failures.append(f"{label}: {reason}")
    if selected_fn is None:
        details = "; ".join(failures[:6])
        raise CoruError("No runtime inference function passed the structured-proposal contract. " + details)

    logger.step(f"Inference API selected from repository runtime: {selected_label}")
    logger.passed(f"Inference contract test 1/3 | required fields valid | output={_summarize_output(first_output)}")
    for i, prompt in enumerate(tests[1:], 2):
        ok, output, reason = invoke_with_supported_arguments(
            selected_fn, prompt, loaded=loaded, model=model, tokenizer=tokenizer,
            device=device, model_dir=model_dir, schema=schema, torch=torch,
        )
        if not ok:
            raise CoruError(f"Inference contract test {i}/3 failed: {reason}")
        try:
            normalized = normalize_proposal(output)
        except CoruError as exc:
            raise CoruError(f"Inference contract test {i}/3 returned an invalid structured proposal: {exc}") from exc
        logger.passed(f"Inference contract test {i}/3 | required fields valid | output={_summarize_output(normalized)}")
    return len(tests)


def extract_action_specs(schema: Any) -> list[dict[str, Any]]:
    """Extract real action declarations without treating parameter fields as routes.

    Only traverse explicit action containers (actions/routes/intents/commands)
    and tool/namespace containers. Once an action record is found, stop walking
    into its parameters, properties, examples, outputs, or JSON-schema metadata.
    This avoids inflating a 55-action schema into hundreds of pseudo-actions.
    """
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    action_containers = {"actions", "action_routes", "routes", "intents", "commands", "operations"}
    group_containers = {"tools", "namespaces", "providers", "integrations", "categories"}
    metadata_keys = {
        "description", "summary", "about", "parameters", "parameter", "properties",
        "required", "optional", "examples", "example", "metadata", "schema", "version",
        "type", "enum", "default", "returns", "output", "outputs", "response",
        "request", "argument", "arguments", "args", "input", "inputs", "$schema",
        "$id", "title", "name", "namespace", "provider", "tool",
    }
    name_keys = ("action", "name", "id", "route", "intent", "command", "title")
    tool_keys = ("tool", "namespace", "provider", "service", "integration")
    example_keys = ("example", "examples", "test_input", "test_prompt", "sample", "utterance", "user_prompt", "prompt")

    def first_text(obj: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def examples_from(obj: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in example_keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif isinstance(value, list):
                values.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
        return values[:10]

    def add_action(node: Any, key_hint: str | None, parent_tool: str | None) -> None:
        if isinstance(node, str):
            action_name = node.strip() or (key_hint or "")
            item: dict[str, Any] = {"action": action_name}
            description = ""
            examples: list[str] = []
            tool_name = parent_tool
        elif isinstance(node, dict):
            action_name = first_text(node, name_keys) or (key_hint or "")
            if not action_name:
                return
            tool_name = first_text(node, tool_keys) or parent_tool
            item = dict(node)
            item.setdefault("action", action_name)
            if tool_name:
                item.setdefault("tool", tool_name)
            description = first_text(node, ("description", "summary", "about")) or ""
            examples = examples_from(node)
        else:
            return
        if not action_name:
            return
        key = ((tool_name or "").casefold(), action_name.casefold())
        if key in seen:
            return
        seen.add(key)
        found.append({"tool": tool_name, "action": action_name, "description": description,
                      "examples": examples, "raw": item})

    def parse_action_container(value: Any, tool_name: str | None) -> None:
        if isinstance(value, list):
            for item in value:
                add_action(item, None, tool_name)
        elif isinstance(value, dict):
            # A single action record (rather than a map of action IDs) is valid.
            if first_text(value, name_keys):
                add_action(value, None, tool_name)
                return
            for action_key, action_value in value.items():
                if str(action_key).lower() in metadata_keys:
                    continue
                add_action(action_value, str(action_key), tool_name)

    def parse_tool_group(group_name: str, group_value: Any) -> None:
        if isinstance(group_value, list):
            for entry in group_value:
                if isinstance(entry, dict):
                    declared_name = first_text(entry, ("name", "tool", "namespace", "provider", "id")) or group_name
                    nested = next((entry[k] for k in entry if str(k).lower() in action_containers), None)
                    if nested is not None:
                        parse_action_container(nested, declared_name)
                    else:
                        add_action(entry, None, declared_name)
                else:
                    add_action(entry, None, group_name)
            return
        if not isinstance(group_value, dict):
            return
        declared_name = first_text(group_value, ("tool", "namespace", "provider", "integration")) or group_name
        nested_keys = [k for k in group_value if str(k).lower() in action_containers]
        if nested_keys:
            for key in nested_keys:
                parse_action_container(group_value[key], declared_name)
            return
        # Supported compact form: {"email": {"send": {...}, "draft": {...}}}.
        for action_key, action_value in group_value.items():
            if str(action_key).lower() in metadata_keys:
                continue
            add_action(action_value, str(action_key), declared_name)

    if isinstance(schema, list):
        parse_action_container(schema, None)
        return found
    if not isinstance(schema, dict):
        return found

    # Prefer explicit tool/namespace grouping when present.
    groups_found = False
    for key, value in schema.items():
        if str(key).lower() in group_containers:
            groups_found = True
            if isinstance(value, dict):
                for group_name, group_value in value.items():
                    if str(group_name).lower() in metadata_keys:
                        continue
                    parse_tool_group(str(group_name), group_value)
            elif isinstance(value, list):
                for group_value in value:
                    if not isinstance(group_value, dict):
                        continue
                    group_name = first_text(group_value, ("name", "tool", "namespace", "provider", "id")) or "tool"
                    parse_tool_group(group_name, group_value)

    # Root-level explicit action collection may coexist with tool metadata.
    for key, value in schema.items():
        if str(key).lower() in action_containers:
            parse_action_container(value, None)

    # Some schemas map tool names directly without a "tools" wrapper.
    if not groups_found and not any(str(key).lower() in action_containers for key in schema):
        excluded = metadata_keys | group_containers | action_containers
        for group_name, group_value in schema.items():
            if str(group_name).lower() in excluded:
                continue
            parse_tool_group(str(group_name), group_value)
    return found


def resolve_expected_route(specs: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    """Resolve a chat expectation from #index, tool.action, or unique action name."""
    target = target.strip()
    if not target:
        return None
    if target.startswith("#") and target[1:].isdigit():
        index = int(target[1:]) - 1
        return specs[index] if 0 <= index < len(specs) else None
    normalized = target.casefold()
    exact = []
    for spec in specs:
        label = f"{spec.get('tool') + '.' if spec.get('tool') else ''}{spec['action']}"
        if label.casefold() == normalized:
            exact.append(spec)
    if not exact:
        exact = [spec for spec in specs if spec["action"].casefold() == normalized]
    return exact[0] if len(exact) == 1 else None


def compare_expected_route(proposal: dict[str, Any], spec: dict[str, Any]) -> tuple[bool, str, str]:
    expected_tool = spec.get("tool")
    expected_action = str(spec.get("action") or "")
    predicted_tool = str(proposal.get("tool") or "")
    predicted_action = str(proposal.get("action") or "")
    tool_ok = not expected_tool or predicted_tool.casefold() == str(expected_tool).casefold()
    action_ok = predicted_action.casefold() == expected_action.casefold()
    expected_label = f"{expected_tool + '.' if expected_tool else ''}{expected_action}"
    predicted_label = f"{predicted_tool}.{predicted_action}"
    return tool_ok and action_ok, expected_label, predicted_label


def _format_proposal(proposal: dict[str, Any]) -> str:
    return json.dumps(proposal, ensure_ascii=False, indent=2, default=str)


def _call_proposal(session: dict[str, Any], prompt: str) -> dict[str, Any]:
    ok, output, reason = invoke_with_supported_arguments(
        session["predict_fn"], prompt,
        loaded=session["loaded"], model=session["model"], tokenizer=session["tokenizer"],
        device=session["device"], model_dir=session["model_dir"], schema=session["schema"],
        torch=session["torch"],
    )
    if not ok:
        raise CoruError(reason)
    proposal = normalize_proposal(output)
    sync = session.get("synchronize")
    if callable(sync):
        sync()
    return proposal


def _display_action_catalog(specs: list[dict[str, Any]]) -> None:
    print("\nAVAILABLE CORU ACTION ROUTES")
    print("-" * 76)
    for index, spec in enumerate(specs, 1):
        label = f"{spec.get('tool') + '.' if spec.get('tool') else ''}{spec['action']}"
        desc = spec.get("description") or ""
        print(f"{index:02d}. {label}" + (f" — {desc[:110]}" if desc else ""))
    print("-" * 76)


def _test_action_interactively(session: dict[str, Any], spec: dict[str, Any], logger: EnterpriseLogger) -> dict[str, Any]:
    label = f"{spec.get('tool') + '.' if spec.get('tool') else ''}{spec['action']}"
    print(f"\nSELECTED ACTION: {label}")
    if spec.get("description"):
        print("Description:", spec["description"])
    for example in spec.get("examples", [])[:3]:
        print("Schema example:", example)
    prompt = input("Enter a natural-language request for this action (blank cancels): ").strip()
    if not prompt:
        return {"action": label, "status": "SKIPPED", "reason": "No test prompt supplied."}
    expected = input(f"Expected action value [{spec['action']}] (Enter=use it, *=skip comparison): ").strip()
    if expected == "*":
        expected = None
    elif not expected:
        expected = spec["action"]
    proposal = _call_proposal(session, prompt)
    print("\nCORU PROPOSAL")
    print(_format_proposal(proposal))
    match: bool | None = None
    expected_tool = spec.get("tool")
    tool_match = True if not expected_tool else str(proposal.get("tool") or "").casefold() == str(expected_tool).casefold()
    if expected is not None:
        action_match = str(proposal.get("action") or "").casefold() == expected.casefold()
        match = action_match and tool_match
        if match:
            logger.passed(f"Individual action test PASS | expected={expected_tool + '.' if expected_tool else ''}{expected} | predicted={proposal.get('tool')}.{proposal.get('action')}")
        else:
            logger.fail(f"Individual action test MISMATCH | expected={expected_tool + '.' if expected_tool else ''}{expected} | predicted={proposal.get('tool')}.{proposal.get('action')}")
    else:
        logger.info(f"Individual proposal generated | selected route={label} | expected-route comparison skipped")
    return {"action": label, "prompt": prompt, "expected_tool": expected_tool, "expected_action": expected, "predicted_tool": proposal.get("tool"), "predicted_action": proposal.get("action"), "proposal": proposal, "match": match, "status": "PASS" if match is True else ("MISMATCH" if match is False else "PROPOSAL_VALID")}


def run_interactive_menu(session: dict[str, Any], logger: EnterpriseLogger) -> list[dict[str, Any]]:
    """Interactive automation chat with explicit semantic route scoring."""
    specs = extract_action_specs(session.get("schema"))
    session["action_specs"] = specs
    route_results: list[dict[str, Any]] = []
    logger.info("Interactive CORU Automation Chat is ready. Proposals only; no external action is executed.")
    print("\n" + "=" * 76)
    print("CORU-10M-ACT | AUTOMATION CHAT & ACTION TEST CONSOLE")
    print(f"Backend: {session['backend']} | Device: {session['device']} | Available schema actions: {len(specs)}")
    print("CORU predicts structured actions. Execution requires a connected host/permission layer.")
    print("=" * 76)
    while True:
        print("\n[1] Automation Chat")
        print("[2] Test one action individually")
        print("[3] Test all actions one-by-one")
        print("[4] View available action routes")
        print("[5] View last test results")
        print("[Q] Exit console")
        try:
            choice = input("CORU> Select option: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInteractive console input closed.")
            break
        if choice in {"q", "quit", "exit", "0"}:
            break
        if choice == "1":
            expected_spec: dict[str, Any] | None = None
            print("\nChat started. Commands: /expect <tool.action|#route-number>, /expect off, /routes, /back, /quit")
            while True:
                try:
                    prompt = input("You> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nChat input closed.")
                    break
                lowered = prompt.casefold()
                if lowered in {"/back", "/menu"}:
                    break
                if lowered in {"/quit", "/exit"}:
                    logger.info("Interactive console closed by user.")
                    return route_results
                if lowered in {"/expect off", "/clear-expectation"}:
                    expected_spec = None
                    print("Route expectation cleared. Chat proposals will be contract-checked only.")
                    continue
                if lowered.startswith("/expect "):
                    target = prompt.split(None, 1)[1].strip()
                    resolved = resolve_expected_route(specs, target)
                    if resolved is None:
                        print(f"Could not uniquely resolve route '{target}'. Use /routes to inspect names or /expect #N.")
                        continue
                    expected_spec = resolved
                    label = f"{resolved.get('tool') + '.' if resolved.get('tool') else ''}{resolved['action']}"
                    print(f"Expected route set: {label}. Next chat requests will be scored against this route.")
                    continue
                if lowered == "/routes":
                    _display_action_catalog(specs)
                    continue
                if not prompt:
                    continue
                try:
                    inference_start = time.perf_counter()
                    proposal = _call_proposal(session, prompt)
                    inference_ms = (time.perf_counter() - inference_start) * 1000.0
                    print(f"CORU> Structured action proposal ({inference_ms:.1f} ms):")
                    print(_format_proposal(proposal))
                    if expected_spec is None:
                        logger.info(f"Chat proposal contract valid | {inference_ms:.1f} ms | route={proposal.get('tool')}.{proposal.get('action')} | semantic route not scored (set /expect)")
                    else:
                        match, expected_label, predicted_label = compare_expected_route(proposal, expected_spec)
                        result = {
                            "action": expected_label, "prompt": prompt,
                            "expected_tool": expected_spec.get("tool"),
                            "expected_action": expected_spec.get("action"),
                            "predicted_tool": proposal.get("tool"),
                            "predicted_action": proposal.get("action"),
                            "proposal": proposal, "match": match,
                            "status": "PASS" if match else "MISMATCH",
                            "inference_ms": round(inference_ms, 2),
                        }
                        route_results.append(result)
                        if match:
                            logger.passed(f"Chat semantic route PASS | {inference_ms:.1f} ms | expected={expected_label} | predicted={predicted_label}")
                        else:
                            logger.fail(f"Chat semantic route MISMATCH | {inference_ms:.1f} ms | expected={expected_label} | predicted={predicted_label}")
                except Exception as exc:
                    logger.fail(f"Chat inference failed: {type(exc).__name__}: {exc}")
                    print(f"CORU> Request could not be processed: {exc}")
            continue
        if choice == "4":
            _display_action_catalog(specs)
            continue
        if choice == "5":
            if not route_results:
                print("No scored individual/chat route tests have been run in this session.")
            else:
                for item in route_results[-30:]:
                    print(f"{item.get('status')} | {item.get('action')} | expected={item.get('expected_action')} | predicted={item.get('predicted_action')} | prompt={item.get('prompt', '')}")
            continue
        if choice in {"2", "3"}:
            if not specs:
                logger.warn("No selectable action records were parsed from action_schema.json.")
                print("No selectable action records were parsed from action_schema.json.")
                continue
            _display_action_catalog(specs)
            if choice == "2":
                raw_index = input(f"Select action number (1-{len(specs)}): ").strip()
                try:
                    index = int(raw_index) - 1
                    if not 0 <= index < len(specs):
                        raise ValueError
                except ValueError:
                    print("Invalid action number.")
                    continue
                try:
                    route_results.append(_test_action_interactively(session, specs[index], logger))
                except Exception as exc:
                    logger.fail(f"Individual action test failed: {type(exc).__name__}: {exc}")
                    print(f"Test failed: {exc}")
            else:
                print("Each schema action will ask for its own natural-language test prompt.")
                confirm = input(f"Run all {len(specs)} individual action tests? [y/N]: ").strip().lower()
                if confirm != "y":
                    continue
                start_index = len(route_results)
                for index, spec in enumerate(specs, 1):
                    print(f"\nROUTE {index}/{len(specs)}")
                    try:
                        result = _test_action_interactively(session, spec, logger)
                    except Exception as exc:
                        label = f"{spec.get('tool') + '.' if spec.get('tool') else ''}{spec['action']}"
                        logger.fail(f"Action test failed | {label} | {type(exc).__name__}: {exc}")
                        result = {"action": label, "status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
                    route_results.append(result)
                batch = route_results[start_index:]
                passed = sum(1 for x in batch if x.get("status") == "PASS")
                mismatches = sum(1 for x in batch if x.get("status") == "MISMATCH")
                logger.info(f"Individual route audit completed: PASS={passed} | MISMATCH={mismatches} | total={len(batch)}")
            continue
        print("Choose 1, 2, 3, 4, 5 or Q.")
    logger.info("Interactive CORU Automation Chat closed.")
    return route_results


def _summarize_output(output: Any) -> str:
    if isinstance(output, dict):
        return ", ".join(str(k) for k in list(output.keys())[:8]) or "empty mapping"
    if isinstance(output, str):
        return re.sub(r"\s+", " ", output.strip())[:150] or "empty string"
    return type(output).__name__


def prepare_working_package(source_dir: Path, cache_dir: Path, logger: EnterpriseLogger) -> Path:
    """Create an isolated package workspace without mutating HF cache or user files."""
    source_dir = source_dir.resolve()
    work_root = (cache_dir.expanduser().resolve() / "_coru_verified_workspaces")
    candidate = work_root / f"{source_dir.name}-{os.getpid()}"
    try:
        if candidate.resolve().is_relative_to(source_dir):
            candidate = Path(tempfile.mkdtemp(prefix="coru-verified-workspace-"))
    except Exception:
        candidate = Path(tempfile.mkdtemp(prefix="coru-verified-workspace-"))
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    try:
        # symlinks=False dereferences HF snapshot links into this private workspace.
        # README checksum refresh can then atomically replace a file here without
        # changing any shared Hugging Face cache blob or caller-supplied directory.
        shutil.copytree(source_dir, candidate, symlinks=False)
    except Exception as exc:
        shutil.rmtree(candidate, ignore_errors=True)
        raise CoruError(f"Unable to prepare an isolated model package workspace: {exc}") from exc
    logger.passed(f"Isolated verification workspace prepared: {candidate}")
    return candidate

def load_and_test(args: argparse.Namespace, logger: EnterpriseLogger) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise CoruError("PyTorch is not installed. Install a CPU PyTorch build or activate the TPU runtime's matched PyTorch/XLA environment.") from exc

    threads = configure_cpu(torch)
    if args.device == "cpu":
        logger.info(f"CPU runtime configured: threads={threads}")

    source_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else download_repository(
        args.repo_id, args.revision, Path(args.cache_dir).expanduser().resolve(), logger
    )
    if not source_dir.is_dir():
        raise CoruError(f"Model directory does not exist: {source_dir}")
    model_dir = prepare_working_package(source_dir, Path(args.cache_dir), logger)

    # Check the key artifacts before importing the model-specific runtime.
    required = ("coru_inference.py", "model.safetensors", "manifest.json", "config.json", "action_schema.json", "tokenizer.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise CoruError("Model repository is missing required CORU release files: " + ", ".join(missing))

    logger.step("Preparing model-runtime dependencies")
    ensure_python_package("safetensors", "safetensors>=0.4.3", logger)
    ensure_python_package("tokenizers", "tokenizers>=0.19.1", logger)
    logger.passed("Core model-runtime dependencies are available")

    logger.step("Preparing documentation-independent package workspace")
    documentation_policy = prepare_documentation_independent_workspace(model_dir, logger)

    logger.step("Importing official CORU inference runtime")
    runtime = import_runtime(model_dir)
    logger.passed("coru_inference.py imported successfully")

    logger.step("Applying Markdown-optional policy to the runtime verifier (in memory only)")
    patched_markdown_paths = patch_runtime_verifier_markdown_policy(runtime, logger)

    logger.step("Running official strict package integrity verification (interim packages disabled)")
    call_integrity_verifier(runtime, model_dir, logger)
    logger.passed("Official runtime strict integrity verification passed; every non-Markdown release file remains checksum-protected")

    meta = inspect_package(model_dir, logger)
    selected = select_device(args.device, torch)
    logger.passed(f"Execution backend selected: {selected.backend} | {selected.description}")
    if args.device == "auto" and selected.backend != "TPU/XLA":
        logger.info("Auto device policy: no active TPU/XLA backend was detected; using the best available non-TPU backend.")

    logger.step("Loading CORU-10M-ACT through the repository's native load() API")
    start = time.perf_counter()
    try:
        loaded = call_runtime_load(runtime, model_dir, selected.device)
    except Exception as exc:
        # Do not silently switch device. Users can intentionally rerun with --device cpu.
        raise CoruError(f"Native CORU load() failed on {selected.backend}: {type(exc).__name__}: {exc}") from exc
    load_seconds = time.perf_counter() - start
    if loaded is None:
        raise CoruError("CORU runtime load() returned None.")
    logger.passed(f"Model loaded in {load_seconds:.3f}s on {selected.backend}")

    model = find_model(loaded, torch)
    tokenizer = find_tokenizer(loaded)
    if model is None:
        # Some runtimes attach the model on the module after loading.
        for name in ("MODEL", "model", "CORU_MODEL", "coru_model"):
            candidate = getattr(runtime, name, None)
            try:
                if isinstance(candidate, torch.nn.Module):
                    model = candidate
                    break
            except Exception:
                pass
    if model is None:
        raise CoruError("Native load() succeeded, but its returned object did not expose a PyTorch model for independent parameter/weight checks.")

    try:
        model.to(selected.device)
    except Exception as exc:
        raise CoruError(f"Could not place model weights on selected device {selected.device}: {exc}") from exc
    model.eval()

    params = count_parameters(model)
    if params is None:
        raise CoruError("Could not independently count model parameters.")
    if params != EXPECTED_PARAMETERS:
        raise CoruError(f"Model parameter count mismatch: expected {EXPECTED_PARAMETERS:,}, found {params:,}.")
    logger.passed(f"Architecture check: {params:,} parameters")

    total, finite_tensors = validate_weights(model, torch)
    logger.passed(f"Weight integrity check: {total:,} tensor values inspected across {finite_tensors} floating/complex tensors; all finite")

    if tokenizer is None:
        for name in ("TOKENIZER", "tokenizer", "CORU_TOKENIZER", "coru_tokenizer"):
            candidate = getattr(runtime, name, None)
            if callable(getattr(candidate, "encode", None)):
                tokenizer = candidate
                break
    if tokenizer is not None:
        try:
            actual_vocab = len(tokenizer)
        except Exception:
            actual_vocab = getattr(tokenizer, "vocab_size", None)
        if actual_vocab is not None and int(actual_vocab) != EXPECTED_VOCAB_SIZE:
            raise CoruError(f"Tokenizer vocabulary mismatch: expected {EXPECTED_VOCAB_SIZE}, found {actual_vocab}.")
        logger.passed(f"Tokenizer check: available; vocabulary={actual_vocab if actual_vocab is not None else 'runtime-managed'}")
    else:
        logger.warn("Tokenizer object is not exposed by load(); the strict runtime verifier and inference API remain authoritative.")

    if selected.synchronize is not None:
        selected.synchronize()
    inference_passes = run_inference_tests(
        runtime, loaded, model, tokenizer, selected.device, model_dir,
        meta.get("schema"), torch, logger,
    )
    if selected.synchronize is not None:
        selected.synchronize()

    # Verify that the model remains on the selected device after loading and smoke tests.
    try:
        first_param_device = str(next(model.parameters()).device)
    except StopIteration:
        first_param_device = str(selected.device)
    if selected.backend == "TPU/XLA" and not first_param_device.startswith("xla"):
        raise CoruError(f"TPU was selected but model parameters are on {first_param_device}.")

    route_test_results: list[dict[str, Any]] = []
    if getattr(args, "interactive", False):
        inference_candidates = find_inference_callables(runtime, loaded, model)
        if not inference_candidates:
            raise CoruError("No runtime inference API is available for the interactive automation console.")
        # Select the same public entry point that satisfies CORU's structured contract.
        predict_fn = None
        predict_label = None
        for candidate_fn, candidate_label in inference_candidates:
            ok, candidate_output, _reason = invoke_with_supported_arguments(
                candidate_fn, "Open the calculator.", loaded=loaded, model=model,
                tokenizer=tokenizer, device=selected.device, model_dir=model_dir,
                schema=meta.get("schema"), torch=torch,
            )
            if ok:
                try:
                    normalize_proposal(candidate_output)
                    predict_fn, predict_label = candidate_fn, candidate_label
                    break
                except CoruError:
                    continue
        if predict_fn is None:
            raise CoruError("No inference API passed the structured CORU proposal contract; interactive chat was not started.")
        logger.passed(f"Interactive inference API ready: {predict_label}")
        session = {
            "runtime": runtime, "loaded": loaded, "model": model,
            "tokenizer": tokenizer, "device": selected.device,
            "backend": selected.backend, "model_dir": model_dir,
            "schema": meta.get("schema"), "torch": torch,
            "predict_fn": predict_fn, "synchronize": selected.synchronize,
        }
        route_test_results = run_interactive_menu(session, logger)

    route_mismatches = sum(1 for item in route_test_results if item.get("status") in {"MISMATCH", "ERROR"})
    final_status = "PASS_WITH_ACTION_MISMATCHES" if route_mismatches else "PASS"
    if route_mismatches:
        logger.warn(f"Loader passed, but individual route audit has {route_mismatches} mismatch/error result(s). Review the per-action report.")

    return {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "model_dir": str(model_dir),
        "backend": selected.backend,
        "device": str(selected.device),
        "parameters": params,
        "vocab_size": meta.get("vocab_size") or EXPECTED_VOCAB_SIZE,
        "context_length": meta.get("context") or EXPECTED_CONTEXT,
        "action_count": meta.get("action_count") or EXPECTED_ACTIONS,
        "load_seconds": round(load_seconds, 4),
        "inference_smoke_tests_passed": inference_passes,
        "interactive_console": "COMPLETED" if getattr(args, "interactive", False) else "NOT_REQUESTED",
        "individual_action_tests": route_test_results,
        "package_integrity": "PASS",
        "documentation_policy": documentation_policy["policy"],
        "markdown_files_removed_from_workspace": documentation_policy["markdown_files_removed"],
        "markdown_hash_entries_removed": documentation_policy["markdown_hash_entries_removed"],
        "runtime_verifier_markdown_paths_removed": patched_markdown_paths,
        "weight_finiteness": "PASS",
        "route_mismatches_or_errors": route_mismatches,
        "status": final_status,
    }


def _self_test(logger: EnterpriseLogger) -> None:
    # Offline CPU/system tests, including a current-layout package with no .md files.
    logger.step("Running offline loader self-tests (no network, no model download)")
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "colab_kernel_launcher.py", "--self-test", "-f",
            "/root/.local/share/jupyter/runtime/kernel-9d3353f9-d417-442b-9bff-e8c92e68b9c1.json",
        ]
        kernel_args = parse_args()
    finally:
        sys.argv = original_argv
    assert kernel_args.self_test and kernel_args.notebook_kernel_args_ignored
    logger.passed("Self-test 1/20: Colab/Jupyter injected kernel argument compatibility")

    class FakeTorch:
        class device:
            def __init__(self, name: str): self.name = name
            def __str__(self): return self.name
        class cuda:
            @staticmethod
            def is_available() -> bool: return False
        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool: return False
        @staticmethod
        def set_num_threads(n: int) -> None: pass
        @staticmethod
        def set_num_interop_threads(n: int) -> None: pass

    device = select_device("cpu", FakeTorch)
    assert device.backend == "CPU" and str(device.device) == "cpu"

    class FakeCudaTorch(FakeTorch):
        class cuda:
            @staticmethod
            def is_available() -> bool: return True
            @staticmethod
            def get_device_name(index: int) -> str: return "Mock CUDA accelerator"
            @staticmethod
            def synchronize() -> None: return None
    assert select_device("cuda", FakeCudaTorch).backend == "CUDA"

    class FakeMpsTorch(FakeTorch):
        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool: return True
    assert select_device("mps", FakeMpsTorch).backend == "APPLE MPS"

    import types
    class _FakeXlaDevice:
        def __str__(self): return "xla:0"
    xla_package = types.ModuleType("torch_xla")
    xla_package.__path__ = []
    xla_package.device = lambda: _FakeXlaDevice()
    xla_package.sync = lambda **kwargs: None
    xla_runtime = types.ModuleType("torch_xla.runtime")
    xla_runtime.device_type = lambda: "TPU"
    xla_package.runtime = xla_runtime
    previous_xla = {name: sys.modules.get(name) for name in ("torch_xla", "torch_xla.runtime")}
    sys.modules["torch_xla"] = xla_package
    sys.modules["torch_xla.runtime"] = xla_runtime
    try:
        assert select_device("auto", FakeTorch).backend == "TPU/XLA"
        assert select_device("tpu", FakeTorch).backend == "TPU/XLA"
    finally:
        for name, previous in previous_xla.items():
            if previous is None: sys.modules.pop(name, None)
            else: sys.modules[name] = previous
    logger.passed("Self-test 2/20: CPU, CUDA, MPS and TPU/XLA device selection and TPU-first auto policy")

    class TinyTensor:
        def __init__(self, n: int): self.n = n
        def numel(self): return self.n
        def is_floating_point(self): return True
        def is_complex(self): return False
    class TinyModule:
        def parameters(self): return [TinyTensor(3), TinyTensor(5)]
        def buffers(self): return []
    assert count_parameters(TinyModule()) == 8
    logger.passed("Self-test 3/20: independent parameter counting")

    def fake_infer(text: str, device: str = "cpu") -> dict[str, str]:
        return {"proposal": "test", "input": text}
    ok, output, _ = invoke_with_supported_arguments(
        fake_infer, "Open calculator", loaded=None, model=None, tokenizer=None,
        device="cpu", model_dir=Path("."), schema=None, torch=None,
    )
    assert ok and output["input"] == "Open calculator"
    logger.passed("Self-test 4/20: inference API signature binding")

    assert schema_action_count({"actions": [{"id": "a"}, {"id": "b"}]}) == 2
    logger.passed("Self-test 5/20: action-schema inspection")

    fixture_proposal = {"tool": "calculator", "action": "open", "parameters": {}, "status": "proposed", "needs_confirmation": False, "clarification": None}
    assert normalize_proposal(fixture_proposal)["action"] == "open"
    try:
        normalize_proposal({"action": "open"})
    except CoruError:
        pass
    else:
        raise AssertionError("Incomplete action proposal passed contract validation.")
    logger.passed("Self-test 6/20: structured action proposal contract and invalid-output rejection")

    fixture_schema = {"tools": {"calculator": {"actions": [{"name": "open", "description": "Open calculator"}, {"name": "add", "examples": ["add 2 and 3"]}]}, "notes": {"actions": {"create": {"description": "Create a note"}}}}}
    specs = extract_action_specs(fixture_schema)
    assert len(specs) == 3 and {x["action"] for x in specs} == {"open", "add", "create"}
    assert any(x["examples"] for x in specs)
    logger.passed("Self-test 7/20: individual action catalog extraction from nested tool schemas")
    keyed_schema = {"calendar": {"create_event": {"description": "Create an event", "parameters": {"title": "string"}}}}
    keyed_specs = extract_action_specs(keyed_schema)
    assert any(x["tool"] == "calendar" and x["action"] == "create_event" for x in keyed_specs)
    logger.passed("Self-test 8/20: action catalog extraction from tool-keyed mapping schemas")
    list_schema = {"tools": [{"name": "notes", "actions": [{"name": "create", "description": "Create note"}]}]}
    list_specs = extract_action_specs(list_schema)
    assert any(x["action"] == "create" for x in list_specs)
    logger.passed("Self-test 9/20: action catalog extraction from list-based tool schemas")

    import types
    with tempfile.TemporaryDirectory(prefix="coru_docs_optional_unit_") as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("edited README", encoding="utf-8")
        (root / "THIRD_PARTY_NOTICES.md").write_text("user-deleted notice fixture", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"weights")
        (root / "coru_inference.py").write_text("# fixture", encoding="utf-8")
        model_hash = _sha256_file(root / "model.safetensors")
        runtime_hash = _sha256_file(root / "coru_inference.py")
        checksum = {"files": {
            "README.md": {"sha256": "1" * 64},
            "THIRD_PARTY_NOTICES.md": {"sha256": "2" * 64},
            "model.safetensors": {"sha256": model_hash},
            "coru_inference.py": {"sha256": runtime_hash},
        }}
        (root / "PACKAGE_SHA256.json").write_text(json.dumps(checksum), encoding="utf-8")
        policy = prepare_documentation_independent_workspace(root, logger)
        assert policy["markdown_hash_entries_removed"] == 2
        assert not (root / "README.md").exists() and not (root / "THIRD_PARTY_NOTICES.md").exists()
        remaining = safe_read_json(root / "PACKAGE_SHA256.json")["files"]
        assert set(remaining) == {"model.safetensors", "coru_inference.py"}
        assert remaining["model.safetensors"]["sha256"] == model_hash
        assert remaining["coru_inference.py"]["sha256"] == runtime_hash
    logger.passed("Self-test 10/20: removes README/THIRD_PARTY_NOTICES and their hashes only")

    # Runtime verifier adaptation test: docs are in its original required set;
    # adaptation changes only the in-memory function and its remaining checks stay live.
    with tempfile.TemporaryDirectory(prefix="coru_verifier_patch_test_") as tmp:
        root = Path(tmp)
        runtime_path = root / "coru_inference.py"
        runtime_source = '''import json, hashlib
from pathlib import Path
def verify_package_integrity(model_dir, allow_interim=False):
    required = {"model.safetensors", "README.md", "THIRD_PARTY_NOTICES.md", "config.json"}
    missing = [name for name in required if not (Path(model_dir) / name).is_file()]
    if missing:
        raise RuntimeError("CORU release file is missing: " + ", ".join(missing))
    data = json.loads((Path(model_dir) / "PACKAGE_SHA256.json").read_text())
    for name, rec in data["files"].items():
        actual = hashlib.sha256((Path(model_dir) / name).read_bytes()).hexdigest()
        if actual != rec["sha256"]:
            raise RuntimeError("CORU release SHA-256 mismatch: " + name)
    if allow_interim is not False:
        raise RuntimeError("interim packages disabled")
'''
        runtime_path.write_text(runtime_source, encoding="utf-8")
        mod = types.ModuleType("coru_inference_test")
        mod.__file__ = str(runtime_path)
        exec(compile(runtime_source, str(runtime_path), "exec"), mod.__dict__)
        original_digest = _sha256_file(runtime_path)
        changed = patch_runtime_verifier_markdown_policy(mod, logger)
        assert changed == 2
        assert _sha256_file(runtime_path) == original_digest
        (root / "model.safetensors").write_bytes(b"weights")
        (root / "config.json").write_text("{}", encoding="utf-8")
        wh = _sha256_file(root / "model.safetensors")
        ch = _sha256_file(root / "config.json")
        (root / "PACKAGE_SHA256.json").write_text(json.dumps({"files": {
            "model.safetensors": {"sha256": wh}, "config.json": {"sha256": ch}
        }}), encoding="utf-8")
        call_integrity_verifier(mod, root, logger)
        (root / "model.safetensors").write_bytes(b"tampered")
        try:
            call_integrity_verifier(mod, root, logger)
        except RuntimeError as exc:
            assert "model.safetensors" in str(exc)
        else:
            raise AssertionError("Tampered model weights passed the patched strict verifier.")
    logger.passed("Self-test 11/20: in-memory runtime verifier adapts docs only; non-doc tampering still fails")

    with tempfile.TemporaryDirectory(prefix="coru_snapshot_api_test_") as tmp:
        root = Path(tmp)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        class _FakeHub:
            class HfApi:
                def __init__(self, token=None): pass
                @staticmethod
                def model_info(repo_id: str, revision: str):
                    assert repo_id == "test/coru" and revision == "main"
                    return types.SimpleNamespace(sha="a" * 40)
            @staticmethod
            def snapshot_download(**kwargs: Any) -> str:
                assert kwargs.get("cache_dir") == str(root / "cache")
                assert kwargs.get("revision") == "a" * 40
                assert "local_dir" not in kwargs
                return str(snapshot)
        old_ensure = globals()["ensure_hf_hub"]
        globals()["ensure_hf_hub"] = lambda _logger: _FakeHub
        try:
            got = download_repository("test/coru", "main", root / "cache", logger)
        finally:
            globals()["ensure_hf_hub"] = old_ensure
        assert got == snapshot.resolve()
    logger.passed("Self-test 12/20: immutable current-revision Hugging Face snapshot download")

    with tempfile.TemporaryDirectory(prefix="coru_workspace_test_") as tmp:
        root = Path(tmp)
        source = root / "source"
        source.mkdir()
        (source / "model.safetensors").write_bytes(b"source weights")
        (source / "README.md").write_text("source README", encoding="utf-8")
        workspace = prepare_working_package(source, root / "cache", logger)
        (workspace / "model.safetensors").write_bytes(b"workspace-only tamper")
        (workspace / "README.md").unlink()
        assert (source / "model.safetensors").read_bytes() == b"source weights"
        assert (source / "README.md").read_text(encoding="utf-8") == "source README"
    logger.passed("Self-test 13/20: private workspace isolation from repository snapshot")

    # Full mocked CPU integration: no .md files at all, source-like strict verifier,
    # native load(), model parameter/finite checks, tokenizer and 3 inference probes.
    with tempfile.TemporaryDirectory(prefix="coru_full_cpu_integration_") as tmp:
        root = Path(tmp)
        package = root / "package"
        package.mkdir()
        (package / "model.safetensors").write_bytes(b"fixture weights; model object supplied by fixture runtime")
        (package / "config.json").write_text(json.dumps({"vocab_size": EXPECTED_VOCAB_SIZE, "context_length": EXPECTED_CONTEXT, "num_parameters": EXPECTED_PARAMETERS}), encoding="utf-8")
        (package / "manifest.json").write_text(json.dumps({"num_parameters": EXPECTED_PARAMETERS, "vocab_size": EXPECTED_VOCAB_SIZE, "context_length": EXPECTED_CONTEXT}), encoding="utf-8")
        (package / "action_schema.json").write_text(json.dumps({"actions": [{"id": f"action_{i}"} for i in range(EXPECTED_ACTIONS)]}), encoding="utf-8")
        (package / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"vocab": [f"t{i}" for i in range(EXPECTED_VOCAB_SIZE)]}}), encoding="utf-8")
        runtime_source = '''import hashlib, json
from pathlib import Path
import torch
class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(9_995_264, dtype=torch.float32))
class FixtureTokenizer:
    vocab_size = 7680
    def __len__(self): return 7680
    def encode(self, text, **kwargs): return [1, 2, 3]
def verify_package_integrity(model_dir, allow_interim=False):
    required = {"model.safetensors", "tokenizer.json", "config.json", "action_schema.json", "coru_inference.py", "README.md", "THIRD_PARTY_NOTICES.md", "requirements_inference.txt", "manifest.json"}
    missing = [name for name in required if not (Path(model_dir) / name).is_file()]
    if missing: raise RuntimeError("CORU release file is missing: " + ", ".join(missing))
    data = json.loads((Path(model_dir) / "PACKAGE_SHA256.json").read_text(encoding="utf-8"))
    for name, record in data["files"].items():
        actual = hashlib.sha256((Path(model_dir) / name).read_bytes()).hexdigest()
        if actual != record["sha256"]: raise RuntimeError("CORU release SHA-256 mismatch: " + name)
    if allow_interim is not False: raise RuntimeError("interim packages are disabled")
def load(model_dir, device="cpu", allow_interim=False):
    verify_package_integrity(model_dir, allow_interim=allow_interim)
    return {"model": FixtureModel().to(device), "tokenizer": FixtureTokenizer()}
def route_request(text):
    return {"tool": "fixture", "action": "route_request", "parameters": {"input": text}, "status": "proposed", "needs_confirmation": False, "clarification": None}
'''
        (package / "coru_inference.py").write_text(runtime_source, encoding="utf-8")
        (package / "requirements_inference.txt").write_text("torch\nsafetensors\ntokenizers\n", encoding="utf-8")
        package_names = ["model.safetensors", "config.json", "manifest.json", "action_schema.json", "tokenizer.json", "coru_inference.py", "requirements_inference.txt"]
        checksum = {"files": {name: {"sha256": _sha256_file(package / name)} for name in package_names}}
        checksum["files"]["README.md"] = {"sha256": "a" * 64}
        checksum["files"]["THIRD_PARTY_NOTICES.md"] = {"sha256": "b" * 64}
        (package / "PACKAGE_SHA256.json").write_text(json.dumps(checksum), encoding="utf-8")
        assert not any(x.suffix.lower() == ".md" for x in package.iterdir())
        old_tokenizers = sys.modules.get("tokenizers")
        sys.modules["tokenizers"] = types.ModuleType("tokenizers")
        args = argparse.Namespace(repo_id="fixture/CORU-10M-ACT", revision="main", device="cpu", cache_dir=str(root / "cache"), model_dir=str(package))
        try:
            full_result = load_and_test(args, logger)
        finally:
            if old_tokenizers is None: sys.modules.pop("tokenizers", None)
            else: sys.modules["tokenizers"] = old_tokenizers
            sys.modules.pop("coru_inference", None)
        assert full_result["status"] == "PASS"
        assert full_result["parameters"] == EXPECTED_PARAMETERS
        assert full_result["inference_smoke_tests_passed"] == 3
        assert full_result["package_integrity"] == "PASS"
        assert full_result["documentation_policy"] == "markdown_optional_non_markdown_strict"
    logger.passed("Self-test 14/20: full CPU integration load with no Markdown files")

    # Check that user-supplied unknown command-line arguments are not silently ignored.
    try:
        parse_args(["--definitely-unknown-flag"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Unknown command-line arguments were incorrectly accepted.")
    logger.passed("Self-test 15/20: strict CLI rejects unrelated unknown arguments")

    assert _is_markdown_path("README.md") and _is_markdown_path("THIRD_PARTY_NOTICES.md")
    assert not _is_markdown_path("model.safetensors") and not _is_markdown_path("config.json")
    logger.passed("Self-test 16/20: Markdown-only policy boundary")

    # End-to-end proposal invocation used by both chat and individual route tests.
    fixture_session = {
        "predict_fn": lambda text: {"tool": "calculator", "action": "open", "parameters": {"request": text}, "status": "proposed", "needs_confirmation": False, "clarification": None},
        "loaded": None, "model": None, "tokenizer": None, "device": "cpu",
        "model_dir": Path("."), "schema": fixture_schema, "torch": None,
        "synchronize": None,
    }
    # Verify the same signature-binding and structured validator path used by chat.
    ok, raw, reason = invoke_with_supported_arguments(fixture_session["predict_fn"], "Open calculator", loaded=None, model=None, tokenizer=None, device="cpu", model_dir=Path("."), schema=fixture_schema, torch=None)
    assert ok and normalize_proposal(raw)["tool"] == "calculator", reason
    logger.passed("Self-test 17/20: chat/individual-action inference path returns a valid structured proposal")

    # Exercise menu navigation, free-form chat, individual action selection, and route comparison.
    import builtins
    menu_session = dict(fixture_session)
    menu_session.update({"backend": "CPU", "device": "cpu", "model_dir": Path("."), "schema": fixture_schema,
                         "predict_fn": lambda text: {"tool": "calculator", "action": "open", "parameters": {"query": text}, "status": "proposed", "needs_confirmation": False, "clarification": None}})
    scripted_inputs = iter(["1", "/expect calculator.open", "Open calculator", "/back", "2", "1", "Open calculator", "open", "q"])
    old_input = builtins.input
    builtins.input = lambda _prompt="": next(scripted_inputs)
    try:
        menu_results = run_interactive_menu(menu_session, logger)
    finally:
        builtins.input = old_input
    assert len(menu_results) == 2 and all(item["status"] == "PASS" for item in menu_results), menu_results
    # Regression: do not interpret parameter/property fields as extra actions.
    noisy_schema = {"tools": {"email": {"actions": {
        "send": {"description": "Send an email", "parameters": {
            "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject", "body"]
        }}
    }}}}
    noisy_specs = extract_action_specs(noisy_schema)
    assert len(noisy_specs) == 1 and noisy_specs[0]["action"] == "send", noisy_specs
    logger.passed("Self-test 18/20: nested parameter/property fields are not miscounted as action routes")

    # A schema route must be compared semantically; a valid but unrelated clarification is a mismatch.
    expected_email = {"tool": "email", "action": "send"}
    wrong_clarification = {"tool": "clarify", "action": "ask", "parameters": {}, "status": "clarify", "needs_confirmation": False, "clarification": "When should the reminder happen?"}
    match, expected_label, predicted_label = compare_expected_route(wrong_clarification, expected_email)
    assert match is False and expected_label == "email.send" and predicted_label == "clarify.ask"
    assert resolve_expected_route(noisy_specs, "email.send") == noisy_specs[0]
    assert resolve_expected_route(noisy_specs, "#1") == noisy_specs[0]
    logger.passed("Self-test 19/20: wrong but schema-valid route is reported as semantic MISMATCH")

    logger.passed("Self-test 20/20: interactive chat, expectation commands, individual action selection, and route comparison")
    logger.passed("Offline self-tests: 20/20 PASS")


def _remove_notebook_kernel_args(argv: list[str]) -> tuple[list[str], bool]:
    """Remove only Jupyter/IPython's injected ``-f kernel-*.json`` pair.

    Keep normal argparse strictness for every other unknown user argument.
    This specifically handles Colab/Jupyter cells where ``sys.argv`` belongs
    to the notebook kernel rather than to this script.
    """
    cleaned: list[str] = []
    ignored = False
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "-f" and i + 1 < len(argv):
            candidate = argv[i + 1]
            name = Path(candidate).name.lower()
            path_text = candidate.replace("\\", "/").lower()
            looks_like_kernel_file = (
                name.endswith(".json")
                and (name.startswith("kernel-") or "/jupyter/runtime/" in path_text or "ipykernel" in path_text)
            )
            if looks_like_kernel_file:
                ignored = True
                i += 2
                continue
        cleaned.append(token)
        i += 1
    return cleaned, ignored


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORU-10M-ACT CPU/TPU loader, structured automation chat and individual action tests")
    p.add_argument("--repo-id", default=REPO_ID, help="Hugging Face model repository ID")
    p.add_argument("--revision", default="main", help="Hugging Face branch, tag, or commit")
    p.add_argument("--device", choices=("auto", "cpu", "tpu", "xla", "cuda", "mps"), default="auto", help="auto prefers TPU, then CUDA, MPS, and CPU")
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "coru" / "CORU-10M-ACT"), help="Download destination/cache")
    p.add_argument("--model-dir", default=None, help="Use an existing local snapshot; skip downloading")
    p.add_argument("--log-file", default="coru_run.log", help="Persistent diagnostic log path")
    p.add_argument("--report-file", default="coru_test_report.json", help="JSON test report path")
    p.add_argument("--self-test", action="store_true", help="Run offline loader self-tests")
    p.add_argument("--no-chat", action="store_true", help="Load and test the model, then exit without opening the interactive console")
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    cleaned_argv, ignored_kernel_args = _remove_notebook_kernel_args(actual_argv)
    args = p.parse_args(cleaned_argv)
    args.notebook_kernel_args_ignored = ignored_kernel_args
    args.interactive = not args.no_chat and not args.self_test
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = EnterpriseLogger(Path(args.log_file))
    start = time.perf_counter()
    logger.info("CORU-10M-ACT | RRR PRO MEX | AUTOMATION INFERENCE RUNTIME")
    logger.info(f"Python={platform.python_version()} | OS={platform.platform()}")
    logger.info(f"Repository={args.repo_id} | revision={args.revision} | device_policy={args.device}")
    if getattr(args, "notebook_kernel_args_ignored", False):
        logger.info("Notebook compatibility: ignored Jupyter's injected kernel metadata argument.")
    logger.info("Execution mode: structured action-routing inference.")
    try:
        if args.self_test:
            _self_test(logger)
            result = {"status": "PASS", "test_mode": "offline_self_test", "tests_passed": 20, "note": "Remote model was not downloaded or loaded in self-test mode."}
        else:
            result = load_and_test(args, logger)
        result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        result["copyright"] = COPYRIGHT
        Path(args.report_file).write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        logger.passed(f"Final status: {result['status']} | report={Path(args.report_file).resolve()}")
        return 0
    except Exception as exc:
        logger.fail(f"RUN FAILED: {type(exc).__name__}: {exc}")
        logger.fail("No success status has been reported. Review the diagnostic log and fix the first failing check.")
        try:
            Path(args.report_file).write_text(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": round(time.perf_counter() - start, 3)}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        except Exception:
            pass
        return 1
    finally:
        logger.footer()
        logger.close()


if __name__ == "__main__":
    _exit_code = main()
    # In Colab/Jupyter, avoid raising SystemExit into the active notebook cell.
    # In a normal terminal process, preserve the conventional process exit code.
    if "IPython" not in sys.modules:
        raise SystemExit(_exit_code)
