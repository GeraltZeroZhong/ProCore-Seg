# ProCore-Seg

ProCore-Seg is a two-stage sparse convolutional network for identifying protein core residues from atomic structures. The project provides a fully reproducible pipeline that starts from publicly available CATH superfamily identifiers, curates atom-level training data with explicit provenance, trains a self-supervised sparse autoencoder followed by a supervised segmentation network, and finishes with inference, evaluation, and visualisation tooling. All tooling is exposed through a single command-line entry point so that end-to-end experiments can be executed with copy-paste commands.

---

## Highlights

- **End-to-end workflow** – Fetch mmCIF/PDB structures, map CATH-derived residue labels via SIFTS, featurise atoms, build HDF5 datasets, train, and evaluate from a single repository.
- **Sparse 3D learning** – Uses MinkowskiEngine sparse tensors to voxelise point clouds lazily and run efficient U-Net style models on protein atom coordinates.
- **Reproducible curation** – Every script records manifests, metadata, and JSON payloads so that downloaded data and generated features can be audited.
- **Reviewer friendly** – Comes with deterministic seeds, logging, environment diagnostics, and metadata-rich checkpoints to simplify replication of published results.

---

## Repository layout

| Path | Description |
| --- | --- |
| `01_data_curation/` | Scripts to download structures, map CATH labels from SIFTS, featurise atoms (element one-hot, SASA, packing density), and orchestrate dataset builds into HDF5. |
| `02_model_architecture/` | Sparse autoencoder and U-Net definitions used for Stage-1 pretraining and Stage-2 segmentation. |
| `03_training/` | Dataloaders, losses, and training loops for both stages, including checkpoint helpers and deterministic utilities. |
| `04_evaluation_inference/` | Inference runners, evaluation metrics, ablation tools, and visualisation scripts (plots, PyMOL overlays, gallery generation). |
| `procore_seg/` | Python package exposing the unified CLI (`procore-seg`), dependency manifest, and environment diagnostics. |
| `README.md` | You are here. |

Additional configuration files (for example YAML configs with CATH identifiers) can be stored alongside the scripts; the tooling accepts explicit CLI arguments so no hidden defaults are required.

---

## Quickstart

### 1. Clone and create an isolated environment

```bash
# From an empty working directory
git clone https://github.com/<your-org>/ProCore-Seg.git
cd ProCore-Seg

# Create and activate a Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install dependencies

```bash
# Core Python dependencies
python -m pip install -r procore_seg/requirements.txt

# Install PyTorch suited to your platform (example: CUDA 12.1 wheels)
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Install MinkowskiEngine (match the CUDA version used above)
pip install -U "git+https://github.com/NVIDIA/MinkowskiEngine@v0.6.1"

# Optional but recommended extras for evaluation/visualisation
pip install matplotlib pandas tabulate fibos
```

> **Tip:** MinkowskiEngine does not publish universal wheels. Consult the [official installation guide](https://github.com/NVIDIA/MinkowskiEngine) if the command above fails for your operating system / CUDA stack.

### 3. Make the repository importable and verify the environment

```bash
# Allow Python to discover the local modules
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run dependency diagnostics (returns non-zero if essentials are missing)
python -m procore_seg.main_cli --log-level INFO doctor

# Snapshot version and environment
python -m procore_seg.main_cli --log-level INFO version
```

The CLI seeds `random`, NumPy, and PyTorch for deterministic runs and prints clear logging when dependencies are missing.

---

## Data curation pipeline

All curation scripts can be invoked directly or via the multiplexed CLI. The examples below assume you are working with the CATH superfamily `1.10.490.10`. Adjust paths and identifiers as required.

```bash
# Common variables
export CATH_ID=1.10.490.10
export RAW_DIR=$PWD/data/raw_pdbs
export WORK_DIR=$PWD/data
mkdir -p "$RAW_DIR" "$WORK_DIR/labels" "$WORK_DIR/features" "$WORK_DIR/processed"
```

### 1. Fetch and cull structures

Downloads mmCIF files that belong to a specific CATH superfamily, storing a manifest of successes/failures and a plain-text list of PDB identifiers.

```bash
python -m procore_seg.main_cli fetch-pdbs \
  --cath-id "$CATH_ID" \
  --out-dir "$RAW_DIR" \
  --max-workers 8 \
  --allow-obsolete false \
  --timeout 20 \
  --retries 3
```

Outputs:

- `data/raw_pdbs/<pdb_id>.cif[.gz]` downloaded structures.
- `data/raw_pdbs/manifest.jsonl` with per-entry metadata (status, error message).
- `data/raw_pdbs/entry_ids.txt` listing the identifiers used downstream.

### 2. Build training-ready HDF5 datasets (recommended)

A single command discovers structures, pulls SIFTS residue labels, runs the atom featuriser, and writes deterministic HDF5 files with a consistent signature. Existing outputs are skipped unless `--overwrite` is provided, so the pipeline is safe to resume.

```bash
python -m procore_seg.main_cli build-dataset \
  --cath-id "$CATH_ID" \
  --raw-dir "$RAW_DIR" \
  --out-dir "$WORK_DIR/processed" \
  --max-workers 8 \
  --sasa-probe 1.4 \
  --sasa-n-points 960 \
  --allow-missing-density
```

For each PDB identifier the pipeline creates `data/processed/<pdb_id>.h5` containing:

- `coords` – `(N, 3)` Cartesian coordinates (Å).
- `features` – `(N, 8)` float32 features ordered `[C, H, O, N, S, Other, SASA, OSP]`.
- `labels` – `(N,)` int64 binary mask for “core” residues.
- `meta` – hierarchical group with provenance (chains, residue indices, download timestamp, software versions).

> **Optional standalone steps:** You can invoke `sifts-map` and `featurize` manually (for debugging or custom preprocessing). The helper `python - <<'PY'` snippet below discovers the preferred structure file and generates intermediate artefacts one entry at a time.
>
> ```bash
> while read -r pdb_id; do
>   python - "$pdb_id" "$RAW_DIR" "$WORK_DIR" "$CATH_ID" <<'PY'
> import sys
> from pathlib import Path
> from subprocess import check_call
>
> pdb_id = sys.argv[1].strip().upper()
> raw_dir = Path(sys.argv[2])
> work_dir = Path(sys.argv[3])
> cath_id = sys.argv[4]
>
> preferred = [".cif.gz", ".cif", ".pdb.gz", ".pdb"]
> structure = None
> for suffix in preferred:
>     candidate = raw_dir / f"{pdb_id.lower()}{suffix}"
>     if candidate.exists():
>         structure = candidate
>         break
> if structure is None:
>     raise SystemExit(f"No structure file found for {pdb_id}")
>
> labels = work_dir / "labels" / f"{pdb_id}.json"
> labels.parent.mkdir(parents=True, exist_ok=True)
> check_call([
>     sys.executable, "-m", "procore_seg.main_cli", "sifts-map",
>     "--pdb-id", pdb_id,
>     "--cath-superfamily", cath_id,
>     "--out", str(labels),
> ])
>
> features = work_dir / "features" / f"{pdb_id}.npz"
> features.parent.mkdir(parents=True, exist_ok=True)
> check_call([
>     sys.executable, "-m", "procore_seg.main_cli", "featurize",
>     "--structure", str(structure),
>     "--labels", str(labels),
>     "--out", str(features),
> ])
> PY
> done < "$RAW_DIR/entry_ids.txt"
> ```
>
> Each `.npz` bundle stores coordinates (`coords`), 8-D features (`features`), binary labels (`labels`), per-atom metadata (`meta`), and quality flags suitable for auditing.

> **Idempotency:** The pipeline skips existing HDF5 files unless `--overwrite` is provided, and it gracefully resumes after interruptions.

---

## Training

All training scripts expect curated HDF5 files produced above. They lazily voxelise coordinates during batching, so you can experiment with different voxel sizes without rebuilding the dataset.

### Stage 1 – Self-supervised sparse autoencoder

Reconstructs quantised point clouds with a density-aware Chamfer distance. Produces `last.pth`, `best.pth`, and `encoder_only.pth` checkpoints enriched with metadata (voxel size, software versions, CLI arguments).

```bash
python -m procore_seg.main_cli pretrain \
  --data-dir "$WORK_DIR/processed" \
  --voxel-size 1.0 \
  --batch-size 4 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 8 \
  --checkpoint-dir "$WORK_DIR/checkpoints/pretrain"
```

Key options:

- `--ids-file`: restrict training to a curated list of PDB IDs.
- `--amp`: enable automatic mixed precision when GPUs are available.
- `--chunk-size`: limit Chamfer loss memory footprint by chunking comparisons.
- `--resume`: continue from an existing checkpoint (retains optimizer, scheduler, scaler state).

### Stage 2 – Supervised segmentation

Initialises a sparse U-Net, optionally loads the pretrained encoder weights, and optimises a density-weighted cross-entropy objective that emphasises dense core regions.

```bash
python -m procore_seg.main_cli segment \
  --data-dir "$WORK_DIR/processed" \
  --voxel-size 1.0 \
  --batch-size 2 \
  --epochs 120 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --encoder-weights "$WORK_DIR/checkpoints/pretrain/encoder_only.pth" \
  --num-workers 8 \
  --checkpoint-dir "$WORK_DIR/checkpoints/segment"
```

Useful flags:

- `--density-feature-idx`: index of the feature channel used as density prior (default 7 → OSP).
- `--dwce-T` and `--dwce-tau`: control the temperature scaling used in density-weighted cross-entropy.
- `--class-weight-pos`: positive class weighting to handle imbalance.
- `--resume`: continue segmentation training from an intermediate checkpoint.

Both training stages respect `--val-split` to carve out validation sets without duplicating data on disk, and they log progress with tqdm progress bars plus structured logging to stdout.

---

## Inference, evaluation, and reporting

### Batch inference

Runs the trained segmentation model over curated HDF5 entries, writing compressed `.npz` artefacts that retain atom-wise logits, probabilities, and metadata required for downstream metrics.

```bash
python -m procore_seg.main_cli infer \
  --model-path "$WORK_DIR/checkpoints/segment/best.pth" \
  --data-dir "$WORK_DIR/processed" \
  --out-dir "$WORK_DIR/inference" \
  --batch-size 4 \
  --workers 4 \
  --voxel-size 1.0 \
  --temperature 1.0
```

Outputs per PDB entry:

- `probs_atom` and `pred_atom`: class probabilities / argmax predictions per atom.
- `labels_atom`: ground-truth labels copied from the HDF5 file for traceability.
- `density_atom`, `coords_atom`, residue identifiers, and provenance metadata.

### Quantitative evaluation

Aggregate metrics (precision/recall, ROC/AUPR, bootstrap confidence intervals) and CSV/Parquet reports from inference artefacts.

```bash
python -m procore_seg.main_cli eval \
  --in-dir "$WORK_DIR/inference" \
  --out-dir "$WORK_DIR/reports" \
  --positive-class 1 \
  --bootstrap 1000
```

The evaluator automatically leverages pandas for rich tables when installed and falls back to pure NumPy otherwise.

### Visualisation & case studies

- **PyMOL overlays:** `python -m procore_seg.main_cli export-pymol --in-dir "$WORK_DIR/inference" --out-dir "$WORK_DIR/pymol"` to generate session files that colour residues by predicted class.
- **Static plots:** `python -m procore_seg.main_cli plots --in-dir "$WORK_DIR/reports"` for publication-ready figures.
- **Interactive gallery:** `python -m procore_seg.main_cli gallery --in-dir "$WORK_DIR/inference" --out "$WORK_DIR/gallery/index.html"` to browse representative examples.

All visualisation commands operate purely on inference outputs; no retraining is required.

---

## Reproducibility & logging

- Global seeds (`--seed`) are applied to Python, NumPy, and PyTorch before each command dispatch.
- Checkpoints bundle CLI arguments and software versions via `build_checkpoint_meta`, easing cross-lab comparisons.
- Every stage writes manifests and metadata next to generated artefacts, ensuring you can trace failures back to source inputs.
- Logging honours `--log-level` and uses consistent ISO timestamps to aid notebook capture and paper appendices.


