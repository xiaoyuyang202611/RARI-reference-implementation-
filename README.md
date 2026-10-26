# RARI — complete reference reimplementation (research prototype)


## What is covered

- Patient-disjoint manifest audit (split and spatial shape checks).
- CT/MRI 2D or `(Z,Y,X)` 3D volumes; no hidden geometry correction.
- A small six-channel 2.5D imaging CNN (preceding/central/following slices,
  current mask, current probability map, bounding-box prompt), inference-visible
  mask/morphology/statistical metadata, and 512-dimensional shared state.
- Action-specific **gain mean**, **0.1/0.5/0.9 conditional quantiles**,
  **binary probability of material loss** for four actions.
- Operational one-step repairability label: `max(0, max_a Dice(M_a,Y)-Dice(M_0,Y))`;
  no-op implicitly has gain zero.
- Auxiliary initial segmentation-quality head (quality-only control) and
  repairability-head ablation (`--no-repairability`).
- Joint multi-task loss: robust mean-gain regression, quantile pinball,
  BCE risk, repairability, quality estimation.
- Budget-aware sequential preserve / repair (max two steps) / refer decisions.
  Risk threshold **estimates** harm and does not guarantee safety.
- Validation-only temperature calibration and **common quality triage** to
  approximately match automatic coverage; risk/gain calibration evaluated on test.
- Held-out evaluation for preserve, fixed-boundary, quality-only,
  `regain_mean_proxy` (***NOT*** original ReGain), distribution-only,
  risk-only, one/two-step RARI, retrospective one/two-step oracles, and
  independently **logged actual ReGain output** if supplied.
- Per-record Dice, gain, population/active-correction material-loss rates,
  any-negative rate, action-call counts, automatic acceptance, by-cohort/phase
  summary, patient-clustered paired bootstrap, risk AUROC/AUPRC/Brier.
- Figure 6 real-image/mask renderer and charting of actual evaluation outputs.
- A synthetic phantom **software test only** (never scientific evidence).

## What this cannot reconstruct from the paper

The manuscript does **not** supply original code/checkpoints, exact action
implementations, full backbone or prompt localization, actual input feature
inventory, final training hyperparameters/loss weights, complete two-step
observed trajectories, matched-budget ReGain call histories or true external
patient masks. We chose reasonable, **explicitly new reference defaults**.
This package does not make the manuscript results reproducible until these
resources are inserted. The demonstration action operators are simple
morphology/intensity heuristics—not MedSAM, nnInteractive, or ReGain.

## Installation (Windows PowerShell / Linux)

Python 3.10+ recommended. In the extracted project directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use a matching CUDA-enabled PyTorch wheel from the official PyTorch instructions
if GPU acceleration is desired; otherwise the code works on CPU.

## End-to-end smoke test on SYNTHETIC phantom arrays

```bash
python -m rari.demo --output demo_data --patients 16
python scripts/check_manifest.py --manifest demo_data/manifest_demo.jsonl --strict-two-step
python -m rari.train --manifest demo_data/manifest_demo.jsonl --output runs/rari.pt --epochs 3 --side 64 --batch-size 4 --device cpu
python -m rari.calibration --manifest demo_data/manifest_demo.jsonl --checkpoint runs/rari.pt --output runs/calibration.json --coverage 0.8
python -m rari.evaluate --manifest demo_data/manifest_demo.jsonl --checkpoint runs/rari.pt --calibration runs/calibration.json --output runs/evaluation
python -m rari.plots --input runs/evaluation --output runs/plots
python -m rari.predict --manifest demo_data/manifest_demo.jsonl --checkpoint runs/rari.pt --calibration runs/calibration.json --case-id SYNTHETIC_015 --output runs/predicted_mask.npy
```

**Demo outputs are not the manuscript's results.** Dataset uses artificial
shapes and demo action operators; there is no medical CT or gastric lesion.

## Real data manifest specification

One JSON object per line of `manifest.jsonl` (paths relative to the manifest):

```json
{
  "case_id":"external_case_001_venous",
  "patient_id":"external_patient_001",
  "cohort":"gastric_multiphase",
  "phase":"venous",
  "split":"test",
  "image":"data/c001/image.npy",
  "initial":"data/c001/initial_mask.npy",
  "gt":"data/c001/tumour_reference.npy",
  "prob":"data/c001/initial_probability.npy",
  "prompt_box":[45,30,160,150],
  "slice_index":48,
  "trajectories":{
    "prompt":"data/c001/actions/prompt.npy",
    "local":"data/c001/actions/local.npy",
    "boundary":"data/c001/actions/boundary.npy",
    "volume":"data/c001/actions/volume.npy",
    "prompt|boundary":"data/c001/actions/prompt_boundary.npy"
  },
  "regain_final":"data/c001/actual_regain_final.npy",
  "regain_calls":1,
  "regain_referred":false
}
```

The example above is **schema illustration only**; no real patient file exists
in this archive. For strict full two-step analysis, save every admissible pair
of interventions (`a|b` for each of the 4×4 sequences); do not add independent
single-step gains to synthesize trajectories. `--strict-two-step` audits that.

- Arrays may be `.npy` (H,W) or (Z,Y,X), and `.nii`/`.nii.gz` (automatically
  transposed from NIfTI X,Y,Z to Z,Y,X). The same image/mask geometry is assumed;
  verify affines/orientations **outside** this code—matching shape is not enough.
- No ground-truth-derived prompts on held-out scans. The precomputed action
  mask files must come from executing each action with inference-available
  inputs, never from the reference masks or hand-drawn desired outputs.
- `gt` may be omitted for individual `rari.predict`, but is essential for
  training and post-hoc evaluation; the state encoder never reads GT.
- A complete bank of candidate actions for each case is strongly recommended.
  Missing actions should not correlate with outcome or domain, to avoid
  evaluation bias. Full two-step oracles need all actual executed trajectories.
- Every scan, slice and phase for one person must stay in the same partition.
- For gastric tumour segmentation, `gt` must be the **tumour** annotation,
  not a stomach-organ mask. Verify diagnosis, phase matching and permissions.
- `regain_final` is an actual precomputed ReGain mask if available. In its
  absence, only the mean-head proxy is evaluated; **never label proxy numbers
  as ReGain experiment results**.

## Real dataset training and validation

```bash
python scripts/check_manifest.py --manifest /path/to/manifest.jsonl --strict-two-step
python -m rari.train --manifest /path/to/manifest.jsonl --output runs/rari_real.pt --epochs 60 --side 96 --batch-size 8
python -m rari.calibration --manifest /path/to/manifest.jsonl --checkpoint runs/rari_real.pt --output runs/real_calibration.json --coverage 0.8
python -m rari.evaluate --manifest /path/to/manifest.jsonl --checkpoint runs/rari_real.pt --calibration runs/real_calibration.json --output runs/real_test --split test
python -m rari.plots --input runs/real_test --output runs/real_plots
```

For an ablation without repairability supervision, run `rari.train` with
`--no-repairability` into a separate checkpoint and use identical splits,
model capacity, seeds, thresholds and action budgets. The full/one-step policy applies a **reconstruction-specific** minimum
predicted repairability gate (default 0.02 Dice) before considering actions.
The manuscript does not supply the original gate formula. The
`--no-repairability` ablation disables this gate and supervision; it is a
*component* ablation, **not** a pure training-supervision-only control.
To isolate supervision specifically, implement a matched untrained head with
identical gating, retrain multiple seeds and compare held-out decisions.
No result of the supplied archive itself establishes hypothesis H1.

The shared triage matches approximately 80% auto-acceptance on validation.
The paper's stronger claim of fair comparison also requires matched model
calls, clinician-review time and exact patient-level outcomes. Current
`regain_logged` uses submitted logged masks; its call counts must be supplied
for an actual cost comparison. Coverage/call equality are **not guaranteed**.

