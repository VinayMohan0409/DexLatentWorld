# DexLatentWorld: 3D Cross-Embodiment Latent Action World Models

Code for the CVPR 2027 submission *DexLatentWorld: 3D Cross-Embodiment Latent Action World Models for Zero-Shot Robotic Manipulation*.

The model learns latent actions from 3D scene particles. The training data is H2O human manipulation, retargeted to eight robot hands.
- **Encoder:** a shared Point Transformer V3.
- **Inverse dynamics:** infers a 4 × 8 latent action from two frames.
- **Forward dynamics:** predicts per-particle motion over five 0.2 s steps.
- **Contrastive loss:** aligns the latents of one transition performed by different hands.

At deployment, CEM searches the latent space directly.

## Layout

| Path | Contents |
|---|---|
| `src/lawm/` | the library. `preprocessing/`, `embodiments/`: Stage 1 data. `data/`, `model/`, `train/`: Stage 2 model and training. `sim/`: simulation (`gripper/`, plus `dexwm/` with `planspace/` and `bench/`) |
| `scripts/` | training entry point (`train_chunked.py`) |
| `experiments/latent_analysis/` | §5.1–5.3: latent conditioning ablation, cross-decode, embodiment probe, human-to-robot transfer |
| `experiments/planspace/` | §5.5: planning and execution ablation (Table 3) |
| `experiments/dexwm_benchmark/` | §5.6: DexWM benchmark (Table 4) |
| `results/` | the reported numbers and the training run record |
| `configs/`, `splits/`, `requirements/` | settings, H2O split, package versions |

The gripper experiment (§5.4) needs no driver: it is a single module, `lawm.sim.gripper.runner`.

## Setup

| Environment | Requirements | Used for |
|---|---|---|
| refinement (Python 3.11, CPU) | `requirements/refinement.txt` | Stage 1 contact refinement |
| model (Python 3.10, CUDA) | `requirements/model.txt`; `requirements/sim.txt` for simulation | training, evaluation, simulation |
| SAM 3 (Python 3.12) | `requirements/sam3.txt` | goal masks (§5.6) |

External code and data are obtained from their original sources, under their own licences: [H2O](https://taeinkwon.com/projects/h2o/), [MANO](https://mano.is.tue.mpg.de/), [UniDex](https://github.com/unidex-ai/UniDex) (commit `97d869e`), [DexWM](https://github.com/facebookresearch/dexwm) (with its robosuite / robocasa forks), [CoTracker3](https://github.com/facebookresearch/co-tracker) and [SAM 3](https://github.com/facebookresearch/sam3). The CoTracker3 and SAM 3 commits are in `requirements/versions.txt`.

Set `PYTHONPATH=src` (plus DexWM's two forks for simulation), `LAWM_DATA_ROOT` (the data folder), `DEXWM_ROOT`, and `COTRACKER_REPO` / `SAM3_REPO` where they are used.

Tests: `pytest tests`.

## Data

```
$LAWM_DATA_ROOT/training_data/h2o_cross_embodiment_pointclouds/<subject>/<session>/<take>/
    000000.npz ...   per frame: hands (9, 2048, 3), hand_valid (9,), object_pose (4, 4)
    sequence.npz     shared scene: table, object, distractors, cameras
$LAWM_DATA_ROOT/training_data/h2o_cross_embodiment_pointclouds_heldout/   validation split + held-out hands
$LAWM_DATA_ROOT/stage3/dexwm/                                             DexWM benchmark demos
```

Embodiments: human, ability, allegro, inspire, leap, oymotion, shadow, wuji, xhand.

Segments: hand, object, table, distractor.

## Checkpoints

Trained checkpoints will be released separately. The configs and defaults expect the pretrained model at `checkpoints/adaln_s0_k4/best.pt`: training writes `best.pt` to `<LAWM_DATA_ROOT>/../runs/<run-name>/`, so copy it there. The fine-tunes in §5.5 and §5.6 are written to the folder given as `$CK`.

## Training

**Stage 1, preprocessing.**
- `lawm.preprocessing.contact_refinement.run`: MANO contact refinement.
- `lawm.preprocessing.regenerate_scene`: table, object and distractors.
- `lawm.preprocessing.export_extra_embodiments`: adds the held-out hands.

**Stage 2, training** (one node, 4 GPUs):
```
torchrun --nproc_per_node 4 scripts/train_chunked.py --run-name adaln_s0_k4 --config configs/h2o_50pct.json --epochs 20 --seed 0 --k 4
```
The full run record is in `results/adaln_s0_k4/`.

## Experiments

Run from the repository root. `--no-amp` is for GPUs without bf16 attention.

### Latent analysis (§5.1–5.3)
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

### Gripper manipulation (§5.4, Table 2)
```
python -m lawm.sim.gripper.runner --config configs/stage3/gripper.json --out runs/gripper --planner lawm \
    --checkpoint checkpoints/adaln_s0_k4/best.pt --manifest results/adaln_s0_k4/run_manifest.json \
    --prior checkpoints/adaln_s0_k4/prior_h2o_subject1.npz
```
The latent prior (mean and spread of the encoder's latents on 400 training windows) is built with `python scripts/build_dexwm_prior.py --checkpoint checkpoints/adaln_s0_k4/best.pt --manifest results/adaln_s0_k4/run_manifest.json --out checkpoints/adaln_s0_k4/prior_h2o_subject1.npz`.
The robot is a Franka Panda in RoboSuite, 10 seeds per task. `--planner pointworld` runs the PointWorld baseline.

### Planning and execution ablation (§5.5, Table 3)
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
Evaluation, per arm and task (`reach`, `grasp`, `place`), on the episodes in `experiments/planspace/episodes/`:
```
python -m lawm.sim.dexwm.planspace.evaluate --arm L --task reach --episodes experiments/planspace/episodes/episodes_reach_20.json --out runs/planspace/reach/L
```
- A1 (A_IK) and A2 (A_dir) add `--g $CK/maps_achieved/g.pt --action-prior $CK/maps_achieved/action_prior.npz`.
- L2A (L_dec) adds `--l2a $CK/maps/l2a_achieved.pt`.
- L (L_IK) needs no map.

Every arm uses the same checkpoint, episodes and CEM budget (64 × 4). Reported results: `results/planspace/`.

### DexWM benchmark (§5.6, Table 4)
The benchmark is the 50 DexWM demos of each task, seen through DexWM's single camera (`robot0_robotview_2`). The goal hand is fitted to the goal image with SAM 3 masks; no goal joint angles are used. Per task `T` in `place`, `grasp`, `reach`:
```
python -m lawm.sim.dexwm.bench.goal_hands model --root runs/goals
python -m lawm.sim.dexwm.bench.goal_frames --out runs/goals --task $T
python -m lawm.sim.dexwm.bench.goal_hands masks --root runs/goals --task $T
python -m lawm.sim.dexwm.bench.goal_hands fit --root runs/goals --task $T --method torch
```
Fine-tune as in §5.5, with single-camera play:
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

## Licences

Research use only. Vendored code keeps its own licence (`LICENSE` and `NOTICE` in each vendored folder). Models trained with this code use data derived from H2O (Kwon et al., ICCV 2021) and MANO (Romero et al., SIGGRAPH Asia 2017) and are for non-commercial research only, under those licences.

| Component | Licence |
|---|---|
| [Utonia](https://github.com/Pointcept/Utonia) PTv3 (`src/lawm/model/vendor/utonia`) | Apache-2.0 |
| robosuite PickPlace (`src/lawm/sim/gripper/vendor`) | MIT |
| DexWM success checks (`src/lawm/sim/dexwm/success`) | CC BY-NC 4.0 |
| MANO / manopth | MANO licence / GPL-3.0 |
| H2O, UniDex, DexWM, robosuite/robocasa forks, SAM 3, CoTracker3 | see upstream |
