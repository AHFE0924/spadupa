# Research Steps Log

## 2026-05-28
- Removed ISEF-specific wording and emojis from the pipeline for a cleaner, professional tone.
- Added sequence clustering utilities with cd-hit support and a Biopython fallback for 30% identity clustering.
- Added GroupKFold cross-validation script with ROC/PR curves and mean/std AUC reporting.
- Added residue-importance script to map embedding-based importance scores onto NDM-1 structure.
- Fixed script import paths for Kaggle execution (groupkfold_cv, cluster_sequences, residue_importance).
- Added GroupKFold guardrails for small cluster counts (write summary and exit cleanly).

## 2026-05-29
- Added UniProt superfamily fetch + clustering script (B1 MBL families, 40% identity).
- Added GroupKFold CV permutation/CI reporting for mean ROC/PR AUC stability.
- Added in silico mutational heatmap generator for single-site substitutions.
- Added external DMS validation script for leakage-free benchmarking.
- Added residue-importance permutation option for embedding-based interpretation.

## 2026-07-12
- Replaced GBSP (fixed-weight graph propagation, zero learned parameters) with a
  small E(n)-equivariant GNN (EGNN) as the default graph scorer. New module:
  `scripts/egnn_model.py` (EGNNLayer/EGNN, plain-torch `index_add_`, no
  torch_geometric dependency).
- EGNN trains on a train-fold-only soft target (fraction of TRAINING homologs
  that vary at each position, `compute_train_variant_frequency`/
  `compute_variant_frequency`) -- never on held-out/curated labels -- and uses
  real Cα coordinates (`get_structure_coords`, straight-chain fallback when no
  PDB) so messages condition on actual 3D distance instead of a fixed
  contact/window graph.
- Kept small + regularized (2 layers, hidden=32, dropout, weight decay, early
  stopping, 5-model ensemble) given `_run_pipeline.py::AdvancedGNN`'s prior
  supervised GAT+GCN attempt overfit to ROC-AUC ~0.5 on the small label counts
  available here.
- `groupkfold_cv.py`, `kaggle_multi_enzyme_real.py`, `kaggle_superfamily_2000.py`
  all take `--scorer {egnn,gbsp}` (default egnn); GBSP kept for direct
  comparison. `compute_scores_from_train`/`compute_scores_from_train_egnn`
  return the same dict shape, so LR ablations, KNN baseline, and CV/plotting
  code are unchanged.
- Renamed `gbsp_*` summary/CSV columns to `primary_*` + added a `scorer`
  column, since a column literally named `gbsp_mean_roc_auc` would be
  misleading once it can hold EGNN numbers. Print statements and plot labels
  now use the active scorer's name instead of a hardcoded "GBSP".
- ESM-2 embeddings and DIAMOND-based clustering/CV splits untouched -- only
  the graph-scoring step changed.
- Left `_run_pipeline.py` as-is (legacy NDM-only monolith, not part of the
  active Kaggle/CV workflow; its own `AdvancedGNN`/`GNNTrainer` were already
  marked unused).
