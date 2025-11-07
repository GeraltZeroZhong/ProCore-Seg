"""Command line interface exposing the end-to-end ProCore-Seg pipeline."""

from __future__ import annotations

import argparse
from importlib import util
from pathlib import Path


MODULES = {
    "fetch": ("procore_seg.01_data_curation.01_fetch_cull_pdbs", "main"),
    "map": ("procore_seg.01_data_curation.02_sifts_label_mapper", "main"),
    "featurize": ("procore_seg.01_data_curation.03_atom_point_featurizer", "main"),
    "build": ("procore_seg.01_data_curation.04_build_dataset_pipeline", "main"),
    "pretrain": ("procore_seg.03_training.train_pretrain", "main"),
    "segment": ("procore_seg.03_training.train_segment", "main"),
    "infer": ("procore_seg.04_evaluation_inference.inference", "infer_labels"),
}


def _load_callable(module_path: str, attr: str):
    try:
        spec = util.find_spec(module_path)
    except (ModuleNotFoundError, ValueError):
        spec = None
    if spec is None:
        package_path = Path(module_path.replace(".", "/") + ".py")
        if not package_path.exists():
            raise ImportError(f"Cannot locate module {module_path}")
        spec = util.spec_from_file_location(module_path, package_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, attr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=MODULES.keys())
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    module_path, attr = MODULES[parsed.command]
    func = _load_callable(module_path, attr)
    if parsed.command == "infer":
        if len(parsed.args) < 2:
            raise SystemExit("infer requires <pdb_path> <weights_path> [device]")
        pdb_path = Path(parsed.args[0])
        weights_path = Path(parsed.args[1])
        device = parsed.args[2] if len(parsed.args) > 2 else "cpu"
        labels = func(pdb_path, weights_path, device)
        print(labels)
    else:
        func(parsed.args)


if __name__ == "__main__":
    main()
