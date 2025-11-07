# ProCore-Seg

ProCore-Seg is a research prototype for density-aware, noise-robust protein domain segmentation.
The project couples a two-stage sparse convolutional pipeline with automated data curation tools
that transform Protein Data Bank (PDB) entries into feature-rich atomic point clouds.

## Repository layout

```
procore_seg/
├── 01_data_curation/      # Fetch PDBs, map SIFTS labels, build HDF5 datasets
├── 02_model_architecture/ # Sparse MinkowskiEngine autoencoder and U-Net definitions
├── 03_training/           # Datasets, custom losses, pretraining and fine-tuning scripts
├── 04_evaluation_inference/ # Metrics, inference helpers, visualisation utilities
├── configs/               # Example YAML configuration files
├── main_cli.py            # Thin command line wrapper for common workflows
└── requirements.txt       # Python dependencies
```

The numbered folders mirror the end-to-end workflow: data curation, self-supervised pretraining,
noise-robust segmentation, and evaluation.

## Quick start

1. Install dependencies:
   ```bash
   pip install -r procore_seg/requirements.txt
   ```
2. Populate a configuration file (an example lives in `procore_seg/configs/`).
3. Run the dataset builder:
   ```bash
   python -m procore_seg.main_cli build procore_seg/configs/cath_1.10.490.10.yaml
   ```
4. Launch self-supervised pretraining:
   ```bash
   python -m procore_seg.main_cli pretrain procore_seg/configs/cath_1.10.490.10.yaml
   ```
5. Fine-tune for segmentation:
   ```bash
   python -m procore_seg.main_cli segment procore_seg/configs/cath_1.10.490.10.yaml
   ```

## License

This code is released for research purposes. Please review dependency licenses before use in
production systems.
