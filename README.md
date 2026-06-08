# Continual Edge Device Personalisation

*A survey of on-device µ-training, anchored with a single-script Raspberry Pi 5 pilot measurement.*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20599083.svg)](https://doi.org/10.5281/zenodo.20599083)

This repository accompanies the IEEE conference paper

> Marty Lauterbach. **Continual µ-Training on Edge Devices: A Survey of On-Device Personalisation.** *Aspekte der Kommunikation, Reutlingen University, 2026.*

It contains the LaTeX source of the paper and a reproducible measurement pipeline that produces the per-update energy data point the paper's central claim is anchored on.

## Headline result

Measured end-to-end on a Raspberry Pi 5 via on-chip Renesas DA9091 PMIC telemetry (`vcgencmd pmic_read_adc`, 5 Hz, 1,728 samples over a 440 s run):

| Configuration | Per-update wall time | Per-update marginal energy |
|---|---|---|
| µ-training (2.06% trainable) | 33.8 ms | 118 mJ |
| Full-network training (100% trainable) | 137.9 ms | 475 mJ |

**µ-training is 4.08× faster and 4.02× more marginal-energy-efficient than full-network training** on the same device, with no thermal throttling and zero PMIC read errors.

## What the pilot script does

A single command on a Raspberry Pi 5:

```bash
cd pilot
pip install -r requirements.txt
python3 pilot_measurement.py
```

Runs ~7 minutes 30 seconds and executes:

0. **PMIC self-test** — verifies `vcgencmd pmic_read_adc` is reachable and readings are plausible.
1. **Idle baseline** (30 s).
2. **Inference micro-benchmark** — FP32 timing + 30 s sustained-load power. INT8 dynamic quantisation attempted and skipped gracefully if the PyTorch ARM build does not support it.
3. **Trainable-fraction sweep** — 5 configurations (last-layer-only, µ-train ~2 %, ~10 %, ~50 %, full); per-update timing + 30 s sustained training-load power for each. Yields the energy-vs-trainable-fraction curve in Fig. 2.
4. **Optimizer comparison** — SGD vs Adam under the µ-training freeze pattern.
5. **Drift detector** — UDDA-TC-style WLV+CUSUM signal layered on continuous inference; reports per-decision compute and energy cost.
6. **BWT proxy** — synthetic two-task forgetting probe scored with the BWT metric of [Lopez-Paz & Ranzato, NeurIPS 2017](https://arxiv.org/abs/1706.08840).
7. **Idle decay** (5 s).

A background thread polls PMIC + SoC temperature + CPU clock at 5 Hz throughout the run; every sample is tagged with the active pipeline phase and written to `pilot_data.csv` (suitable for pandas / matplotlib analysis).

To regenerate the paper figures from the JSON + CSV:

```bash
python3 make_figures.py
```

## Building the paper

```bash
pdflatex Lauterbach_OnDevicePersonalisation.tex
pdflatex Lauterbach_OnDevicePersonalisation.tex     # second pass for cross-refs
```

## Honest caveats

- **PMIC scope:** the Pi 5 PMIC measures the secondary rails it generates (3V3, 1V8, VDD\_CORE, DDR, …) but not the 5 V main-rail current. Power going from 5 V to USB peripherals, HATs, NVMe, or the fan is excluded from `total_W`. Cross-phase comparisons remain valid because the unmeasured offset is approximately constant across phases.
- **Hardware tier:** the Pi 5 is *not* a Cortex-M-class MCU. The measured per-update energies are in mJ; the paper's proposed Cortex-M target ceiling is on the order of 10² µJ — three to four orders of magnitude lower. The Pi 5 measurement sits at the same hardware tier the surveyed µ-training works actually deploy on (Raspberry Pi 3, Radxa Zero); repeating the sweep on Cortex-M4F/M7 hardware is the natural next step.
- **BWT proxy:** the two-task forgetting probe runs on synthetic data with a deterministic input→label mapping plus mean shift and label rotation between tasks. It demonstrates the methodology and reproduces catastrophic forgetting (BWT = −1.0 for full-network training) but is not a substitute for a real biosignal continual-learning benchmark.
- **INT8 path:** `torch.ao.quantization.quantize_dynamic` quantises Linear layers only; Conv1d layers in this model remain FP32, so the realisable speedup is bounded by the Linear-layer fraction. On the PyTorch 2.12 ARM build used for this measurement, the dynamic-quantisation code path failed at instantiation; the run logs the failure and continues.

## Author

Marty Lauterbach, Reutlingen University.
ORCID: [0009-0007-3396-872X](https://orcid.org/0009-0007-3396-872X).

```bibtex
@misc{lauterbach2026continual,
  title     = {Continual µ-Training on Edge Devices: A Survey of On-Device Personalisation},
  author    = {Lauterbach},
  year      = {2026},
  doi       = {10.5281/zenodo.20599083},
  url       = {https://doi.org/10.5281/zenodo.20599083}
}
```
