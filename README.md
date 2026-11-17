# DexLatentWorld: 3D Cross-Embodiment Latent Action World Models for Zero-Shot Robotic Manipulation

Vinay Udit Mohan, Xuchuan Huang, Rolandos Alexandros Potamias  
Imperial College London

[Project page](https://vinaymohan0409.github.io/DexLatentWorld/) · Paper (coming soon)

DexLatentWorld learns continuous latent actions from 3D scene particles of human manipulation retargeted to eight robot hands. A shared Point Transformer V3 encoder feeds an inverse dynamics model, which infers a 4 × 8 latent action from two frames, and a forward dynamics model, which predicts per-particle motion over five 0.2 s steps. A cross-embodiment contrastive loss aligns the latents of one transition performed by different hands. At deployment, CEM searches the latent space directly and robot targets are recovered from the predicted particles using known geometry and kinematics.

## Citation

If you find this work useful, please cite:

```bibtex
@article{dexlatentworld2026,
  title   = {DexLatentWorld: 3D Cross-Embodiment Latent Action World
             Models for Zero-Shot Robotic Manipulation},
  author  = {Mohan, Vinay Udit and Huang, Xuchuan and
             Potamias, Rolandos Alexandros},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Table of contents

- [Setup](#setup)
- [Data](#data)
- [Checkpoints](#checkpoints)
- [Training](#training)
- [Reproducing the paper](#reproducing-the-paper)
  - [Latent analysis (Sec. 5.1–5.3)](#latent-analysis-sec-5153)
  - [Zero-shot gripper manipulation (Sec. 5.4)](#zero-shot-gripper-manipulation-sec-54)
  - [Planning and execution ablation (Sec. 5.5)](#planning-and-execution-ablation-sec-55)
  - [DexWM benchmark (Sec. 5.6)](#dexwm-benchmark-sec-56)
- [Acknowledgements](#acknowledgements)
- [Repository structure](#repository-structure)

## Setup

| Environment | Requirements | Used for |
|---|---|---|
| refinement (Python 3.11, CPU) | `requirements/refinement.txt` | hand-trajectory refinement |
| model (Python 3.10, CUDA) | `requirements/model.txt`; `requirements/sim.txt` for simulation | training, evaluation, simulation |
| SAM 3 (Python 3.12) | `requirements/sam3.txt` | goal-hand masks (Sec. 5.6) |

External code and data are obtained from their original sources, under their own licences (see [Acknowledgements](#acknowledgements)). The CoTracker3 and SAM 3 commits are in `requirements/versions.txt`.

Set `PYTHONPATH=src` (plus DexWM's `robot_sim_dexwm/robosuite_murp` and `robot_sim_dexwm/robocasa-murp` for simulation), `LAWM_DATA_ROOT` (the data folder), `DEXWM_ROOT`, and `COTRACKER_REPO` / `SAM3_REPO` where they are used.

Tests: `pytest tests`.

## Data

```
$LAWM_DATA_ROOT/training_data/h2o_cross_embodiment_pointclouds/<subject>/<session>/<take>/
    000000.npz ...   per frame: hands (9, 2048, 3), hand_valid (9,), object_pose (4, 4)
    sequence.npz     shared scene: table, object, distractors, cameras
$LAWM_DATA_ROOT/training_data/h2o_cross_embodiment_pointclouds_heldout/   validation split + held-out hands
$LAWM_DATA_ROOT/stage3/dexwm/                                             DexWM benchmark demos
```

Embodiments: human, Ability, Allegro, Inspire, LEAP, OYMotion, Shadow, Wuji, XHand. Held-out hands: SVH, DG-5F, Dex3. Segments: hand, object, table, distractor. The train/validation split is by sequence (`splits/h2o_50pct.json`).

## Checkpoints

Trained checkpoints will be released separately. The configs and defaults expect the pretrained model at `checkpoints/adaln_s0_k4/best.pt`: training writes `best.pt` to `<LAWM_DATA_ROOT>/../runs/<run-name>/`, so copy it there. The fine-tunes in Sec. 5.5 and 5.6 are written to the folder given as `$CK`.

## Training

**Stage 1, preprocessing.**
- `lawm.preprocessing.contact_refinement.run`: hand-trajectory refinement.
- `lawm.preprocessing.regenerate_scene`: table, object and distractors.
- `lawm.preprocessing.export_extra_embodiments`: adds the held-out hands.

**Stage 2, training** (one node, 4 GPUs):
```
torchrun --nproc_per_node 4 scripts/train_chunked.py --run-name adaln_s0_k4 --config configs/h2o_50pct.json --epochs 20 --seed 0 --k 4
```
The full run record is in `results/adaln_s0_k4/`.

## Reproducing the paper

Each subsection gives the commands that produce one set of results in the paper. Run them from the repository root. `--no-amp` is for GPUs without bf16 attention.

### Latent analysis (Sec. 5.1–5.3)
```
python experiments/latent_analysis/ablate_segments.py --out runs/table1 --no-amp       # Table 1
python experiments/latent_analysis/report.py runs/table1
python experiments/latent_analysis/cross_decode.py --config configs/eval/cross_decode_val.json --out runs/cross_decode --no-amp
python experiments/latent_analysis/plot_cross_decode_paper.py --run runs/cross_decode --which mean    # Fig. 5
python experiments/latent_analysis/latent_analysis.py extract --out runs/probe                        # embodiment probe
python experiments/latent_analysis/latent_analysis.py analyze --out runs/probe
python experiments/latent_analysis/latent_transfer_demo.py {links,run,plot} --out runs/pour           # Fig. 6, human-to-robot pour
```
Reported results: `results/latent_analysis/`.

### Zero-shot gripper manipulation (Sec. 5.4)
```
python -m lawm.sim.gripper.runner --config configs/stage3/gripper.json --out runs/gripper --planner lawm \
    --checkpoint checkpoints/adaln_s0_k4/best.pt --manifest results/adaln_s0_k4/run_manifest.json \
    --prior checkpoints/adaln_s0_k4/prior_h2o_subject1.npz
```
The latent prior (mean and spread of the encoder's latents on 400 training windows) is built with `python scripts/build_dexwm_prior.py --checkpoint checkpoints/adaln_s0_k4/best.pt --manifest results/adaln_s0_k4/run_manifest.json --out checkpoints/adaln_s0_k4/prior_h2o_subject1.npz`.
The robot is a Franka Panda in RoboSuite, 10 seeds per task. `--planner pointworld` runs the PointWorld baseline.

### Planning and execution ablation (Sec. 5.5)
Fine-tune on exploratory play from the 13 non-evaluation kitchens. `CK` is the output folder.
```
python -m lawm.sim.dexwm.planspace.play --out runs/play --per-scene 12 --kinds explore --scene production
python -m lawm.sim.dexwm.planspace.tracks --play runs/play --kinds explore
python -m lawm.sim.dexwm.planspace.labels --play runs/play --out runs/index.npz --index-only --clips-per-scene 12
python -m lawm.sim.dexwm.planspace.finetune_wm --stage 1 --labels runs/index.npz --play runs/play --out $CK \
    --steps 3000 --accum 4 --check-every 100 --patience 3 --min-delta 0.5 --check-windows 100
export PLANSPACE_CHECKPOINT=$CK/stage1.pt PLANSPACE_PRIOR=$CK/prior.npz
python -m lawm.sim.dexwm.planspace.labels --play runs/play --out $CK/encoder_latents.npz --clips-per-scene 12 --batch 12
python -m lawm.sim.dexwm.planspace.prior_from_labels --labels $CK/encoder_latents.npz --checkpoint $CK/stage1.pt --out $CK/prior.npz
```
Maps between hand actions and latents, trained on the achieved hand motion:
```
python -m lawm.sim.dexwm.planspace.achieved_labels --labels $CK/encoder_latents.npz --play runs/play --out $CK/encoder_latents_achieved.npz
python -m lawm.sim.dexwm.planspace.train_g --labels $CK/encoder_latents_achieved.npz --play runs/play --out $CK/maps_achieved
python -m lawm.sim.dexwm.planspace.action_prior --labels $CK/encoder_latents_achieved.npz --out $CK/maps_achieved/action_prior.npz
python -m lawm.sim.dexwm.planspace.train_l2a_achieved --labels $CK/encoder_latents.npz --play runs/play --out $CK/maps
```
Evaluation, per configuration and task (`reach`, `grasp`, `place`), on the episodes in `experiments/planspace/episodes/`:
```
python -m lawm.sim.dexwm.planspace.evaluate --arm L --task reach --episodes experiments/planspace/episodes/episodes_reach_20.json --out runs/planspace/reach/L
```
- A1 (A_IK) and A2 (A_dir) add `--g $CK/maps_achieved/g.pt --action-prior $CK/maps_achieved/action_prior.npz`.
- L2A (L_dec) adds `--l2a $CK/maps/l2a_achieved.pt`.
- L (L_IK) needs no map.

Every configuration uses the same checkpoint, episodes and CEM budget (64 × 4). Reported results: `results/planspace/`.

### DexWM benchmark (Sec. 5.6)
The benchmark is the 50 DexWM demos of each task, seen through DexWM's single camera (`robot0_robotview_2`). The goal hand is fitted to the goal image with SAM 3 masks; no goal joint angles are used. Per task `T` in `place`, `grasp`, `reach`:
```
python -m lawm.sim.dexwm.bench.goal_hands model --root runs/goals
python -m lawm.sim.dexwm.bench.goal_frames --out runs/goals --task $T
python -m lawm.sim.dexwm.bench.goal_hands masks --root runs/goals --task $T
python -m lawm.sim.dexwm.bench.goal_hands fit --root runs/goals --task $T --method torch
```
Fine-tune as in Sec. 5.5, with single-camera play:
```
python -m lawm.sim.dexwm.planspace.play --out runs/play_dexwm --per-scene 12 --kinds explore --scene production --cameras dexwm --floor-above-object 0.02
```
Check the inputs, evaluate and summarise:
```
python -m lawm.sim.dexwm.bench.check_fair --goal-root runs/goals --task $T --play runs/play_dexwm --checkpoint-dir $CK
python -m lawm.sim.dexwm.planspace.evaluate --arm L --task $T --episodes experiments/dexwm_benchmark/episodes_${T}_50.json \
    --strict runs/goals --allow-flagged-goals --out runs/bench/${T}_L
python -m lawm.sim.dexwm.bench.summary --out runs/bench/${T}_L --task $T --goal-root runs/goals
```
Reported results: `results/dexwm_benchmark/`.

## Acknowledgements

This work builds on the following projects. Code included in this repository keeps its own licence (`LICENSE` and `NOTICE` in its folder); everything else is obtained from the original source.

| Project | Use in this repository | Licence |
|---|---|---|
| [Utonia](https://github.com/Pointcept/Utonia) / [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) | encoder backbone, included in `src/lawm/model/vendor/utonia` | Apache-2.0 |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | gripper tasks; PickPlace included in `src/lawm/sim/gripper/vendor` | MIT |
| [DexWM](https://github.com/facebookresearch/dexwm) | dexterous benchmark; success checks included in `src/lawm/sim/dexwm/success` | CC BY-NC 4.0 |
| [H2O](https://taeinkwon.com/projects/h2o/) | source manipulation data | dataset terms |
| [MANO](https://mano.is.tue.mpg.de/) / [manopth](https://github.com/hassony2/manopth) | hand model | MANO licence / GPL-3.0 |
| [UniDex](https://github.com/unidex-ai/UniDex) (commit `97d869e`) | robot-hand retargeting | see upstream |
| [RoboCasa](https://github.com/robocasa/robocasa) | kitchen scenes (via DexWM's forks) | see upstream |
| [CoTracker3](https://github.com/facebookresearch/co-tracker) | play-data correspondences | see upstream |
| [SAM 3](https://github.com/facebookresearch/sam3) | goal-hand masks | see upstream |
| [PointWorld](https://github.com/NVlabs/PointWorld) | gripper baseline | see upstream |

For research use only. Models trained with this code use data derived from H2O and MANO and are for non-commercial research only, under those licences.

## Repository structure

```
├── configs/                    training, evaluation and simulation settings
│   ├── h2o_50pct.json          training configuration
│   ├── embodiments.json        hand registry (trained and held-out)
│   ├── hands/                  held-out hand specifications
│   ├── eval/                   cross-decode evaluation
│   └── stage3/                 gripper and DexWM simulation settings
├── splits/                     H2O train/validation split
├── requirements/               package versions per environment
├── scripts/                    training entry point, latent prior
├── src/lawm/
│   ├── preprocessing/          Stage 1: H2O readers, hand refinement, retargeting, scene geometry
│   ├── embodiments/            robot-hand specifications and calibration
│   ├── data/                   windows, sampling, normalisation
│   ├── model/                  PTv3 backbone, inverse and forward dynamics, losses
│   ├── train/                  trainer, evaluation, reproducibility
│   ├── eval/                   cross-decode protocol
│   ├── viz/                    hand geometry for figures
│   └── sim/
│       ├── gripper/            Sec. 5.4 tasks, planner, PointWorld baseline
│       └── dexwm/              DexWM environment, planning, control, success checks
│           ├── planspace/      play data, tracking, fine-tuning, maps, evaluation
│           └── bench/          goal frames, goal-hand fitting, input checks, summary
├── experiments/
│   ├── latent_analysis/        Sec. 5.1–5.3 scripts
│   ├── planspace/episodes/     Sec. 5.5 evaluation episodes
│   └── dexwm_benchmark/        Sec. 5.6 evaluation episodes
├── results/                    reported numbers and the training run record
└── tests/                      unit tests
```
