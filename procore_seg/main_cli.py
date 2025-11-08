"""Top-level command-line entrypoint for ProCore-Seg workflows.

This CLI multiplexes data curation, training, inference, evaluation, and
visualisation utilities under a single command. Typical usage examples::

    python -m procore_seg.main_cli fetch-pdbs --help
    procore-seg pretrain --config configs/train.yaml

All subcommands accept additional arguments which are forwarded verbatim to
underlying tools. Each subcommand is lazily imported to keep startup overhead
minimal.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import platform
import random
import subprocess
import sys
from typing import Dict, Optional, Sequence

from . import VERSION

_LOGGER = logging.getLogger("procore_seg.cli")


_COMMAND_MODULES: Dict[str, str] = {
    "fetch-pdbs": "procore_seg.01_data_curation.01_fetch_cull_pdbs",
    "sifts-map": "procore_seg.01_data_curation.02_sifts_label_mapper",
    "featurize": "procore_seg.01_data_curation.03_atom_point_featurizer",
    "build-dataset": "procore_seg.01_data_curation.04_build_dataset_pipeline",
    "pretrain": "procore_seg.03_training.train_pretrain",
    "segment": "procore_seg.03_training.train_segment",
    "infer": "procore_seg.04_evaluation_inference.inference.run_inference",
    "export-pymol": "procore_seg.04_evaluation_inference.inference.export_pymol",
    "eval": "procore_seg.04_evaluation_inference.evaluation.evaluate_dataset",
    "ablate": "procore_seg.04_evaluation_inference.evaluation.ablations",
    "baselines": "procore_seg.04_evaluation_inference.evaluation.baselines",
    "report": "procore_seg.04_evaluation_inference.evaluation.report",
    "plots": "procore_seg.04_evaluation_inference.viz.plots",
    "gallery": "procore_seg.04_evaluation_inference.viz.case_gallery",
    "overlay": "procore_seg.04_evaluation_inference.viz.volume_overlay",
}

_DIAGNOSTICS: Sequence[tuple[str, str, bool]] = (
    ("python", "sys", True),
    ("numpy", "numpy", True),
    ("torch", "torch", True),
    ("MinkowskiEngine", "MinkowskiEngine", False),
    ("h5py", "h5py", True),
    ("biopython", "Bio", True),
    ("requests", "requests", False),
    ("tqdm", "tqdm", True),
    ("PyYAML", "yaml", True),
    ("matplotlib", "matplotlib", False),
    ("pandas", "pandas", False),
    ("fibos", "fibos", False),
)


def log_init(level: str) -> logging.Logger:
    """Initialise deterministic logging for the CLI."""

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    _LOGGER.debug("Logger initialised at level %s", level)
    return _LOGGER


def set_seed(seed: int) -> None:
    """Set seeds across supported libraries to encourage determinism."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - optional dependency
        _LOGGER.debug("NumPy not available when seeding", exc_info=True)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - backend specific
            _LOGGER.debug("Unable to configure cuDNN for determinism", exc_info=True)
    except Exception:  # pragma: no cover - optional dependency
        _LOGGER.debug("PyTorch not available when seeding", exc_info=True)


def dispatch(module_path: str, extra_argv: Sequence[str]) -> int:
    """Dispatch a subcommand to the requested module."""

    _LOGGER.debug("Dispatching to module %s with args %s", module_path, list(extra_argv))
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        _LOGGER.info("Falling back to subprocess for %s", module_path)
        _LOGGER.debug("ImportError when loading %s", module_path, exc_info=True)
        return _run_module_subprocess(module_path, extra_argv)

    main_func = getattr(module, "main", None)
    if not callable(main_func):
        _LOGGER.info("Module %s has no callable main; using subprocess fallback", module_path)
        return _run_module_subprocess(module_path, extra_argv)

    argv_backup = sys.argv[:]
    sys.argv = [module_path.split(".")[-1], *extra_argv]
    try:
        result = main_func()
    finally:
        sys.argv = argv_backup
    if isinstance(result, int):
        return result
    return 0


def _run_module_subprocess(module_path: str, extra_argv: Sequence[str]) -> int:
    cmd = [sys.executable, "-m", module_path, *list(extra_argv)]
    _LOGGER.debug("Executing subprocess: %s", cmd)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def _format_status(name: str, version: Optional[str], ok: bool) -> str:
    status = "OK" if ok else "MISSING"
    if ok:
        detail = version or "unknown"
    else:
        detail = "not-installed"
    return f"{name:<18} : {status:<8} {detail}"


def _check_import(module_name: str) -> tuple[bool, Optional[str], Optional[BaseException]]:
    if module_name == "sys":
        return True, platform.python_version(), None
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import side effect dependent
        return False, None, exc
    version = getattr(module, "__version__", None)
    if module_name == "Bio":
        try:
            import Bio

            version = getattr(Bio, "__version__", None)
        except Exception:
            version = None
    if module_name == "yaml":
        version = getattr(module, "__version__", None)
    return True, version, None


def doctor() -> int:
    """Inspect the runtime environment and report dependency health."""

    print("== ProCore-Seg Environment Doctor ==")
    missing_essentials = False
    torch_module = None
    for display_name, module_name, essential in _DIAGNOSTICS:
        ok, version, err = _check_import(module_name)
        if module_name == "torch" and ok:
            torch_module = importlib.import_module(module_name)
        if not ok and essential:
            missing_essentials = True
        if not ok and err is not None:
            version_text = f"error: {err.__class__.__name__}"  # pragma: no cover - informative only
        else:
            version_text = version
        print(_format_status(display_name, version_text, ok))

    if torch_module is not None:
        cuda_available = torch_module.cuda.is_available()
        print(f"CUDA available     : {cuda_available}")
        if cuda_available:
            try:
                devices = [
                    torch_module.cuda.get_device_name(idx)
                    for idx in range(torch_module.cuda.device_count())
                ]
                for idx, name in enumerate(devices):
                    print(f"GPU[{idx}]            : {name}")
            except Exception:  # pragma: no cover - depends on runtime
                print("GPU details       : error retrieving device names")
    else:
        print("CUDA available     : torch not installed")

    try:
        import MinkowskiEngine  # type: ignore

        print("MinkowskiEngine    : import succeeded")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"MinkowskiEngine    : not available ({exc.__class__.__name__})")

    return 0 if not missing_essentials else 1


def _default_env_snapshot() -> Dict[str, str]:
    snapshot: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy as np

        snapshot["numpy"] = np.__version__
    except Exception:
        snapshot["numpy"] = "not-installed"
    try:
        import torch

        snapshot["torch"] = torch.__version__  # type: ignore[attr-defined]
        snapshot["cuda"] = str(torch.cuda.is_available())  # type: ignore[attr-defined]
    except Exception:
        snapshot["torch"] = "not-installed"
        snapshot["cuda"] = "False"
    try:
        import MinkowskiEngine as me  # type: ignore

        snapshot["minkowski"] = getattr(me, "__version__", "unknown")
    except Exception:
        snapshot["minkowski"] = "not-installed"
    return snapshot


def version() -> int:
    """Print project version and environment snapshot."""

    print(f"ProCore-Seg version : {VERSION}")
    git_hash = _git_short_hash()
    if git_hash:
        print(f"Git commit          : {git_hash}")
    snapshot = _load_env_snapshot()
    print("Environment snapshot:")
    for key, value in snapshot.items():
        print(f"  {key}: {value}")
    return 0


def _git_short_hash() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except Exception:  # pragma: no cover - git optional
        return None
    output = completed.stdout.strip()
    return output or None


def _load_env_snapshot() -> Dict[str, str]:
    try:
        utils_mod = importlib.import_module(
            "procore_seg.04_evaluation_inference.common.utils"
        )
        env_snapshot = getattr(utils_mod, "env_snapshot")
    except Exception:
        env_snapshot = _default_env_snapshot
    snapshot = env_snapshot()
    ordered = {key: snapshot[key] for key in sorted(snapshot.keys())}
    return ordered


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="procore-seg",
        description="Unified command-line interface for ProCore-Seg workflows.",
        add_help=True,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity for the CLI and dispatched tools.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed applied to random, numpy, and torch.",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="Run environment diagnostics.")
    subparsers.add_parser("version", help="Print version and environment snapshot.")

    for command, module_path in _COMMAND_MODULES.items():
        subparsers.add_parser(command, help=f"Dispatch to {module_path}.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, extra = parser.parse_known_args(argv)

    logger = log_init(args.log_level)
    logger.debug("Parsed arguments: args=%s extra=%s", args, extra)
    set_seed(args.seed)

    command = args.command
    if command is None:
        parser.print_help()
        return 2

    if command == "doctor":
        return doctor()
    if command == "version":
        return version()

    module_path = _COMMAND_MODULES.get(command)
    if module_path is None:
        parser.print_help()
        return 2
    exit_code = dispatch(module_path, extra)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
