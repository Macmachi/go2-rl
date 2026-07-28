# Go2 RL — teaching a quadruped robot from scratch, on a consumer GPU

**🌐 Language / Langue : [🇫🇷 Français](README.md) · 🇬🇧 English (this page)**

Reinforcement-learning training of a **Unitree Go2** (bare robot, **no arm, no camera**)
in the **[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)** simulator, on an
**AMD Radeon RX 9070 GPU (ROCm)**. The goal: produce an educational video showing the
robot **learning several skills across checkpoints** — from initial flailing to walking,
running, lying down / getting up, sitting, avoiding obstacles with LiDAR, etc. — with,
alongside it, **the learning curve going up** and a **"real GPU time ↔ months of
robot experience"** equivalence.

**🎬 Preview** — the Go2 learns to walk (robot on the left, *time before falling* on the
right, climbing as training progresses):

![The Go2 learns to walk, next to its learning curve](gifs/marche.gif)

> Animated preview (real-time excerpt). One **GIF per skill** is available in
> [`gifs/`](gifs/). The high-resolution **videos** AND the **checkpoints** (`videos/`, `logs/`)
> are **not versioned** (large, git-ignored): on a fresh clone you recreate them by
> **training** (the code is provided) **then** rendering with `pilot_walk_video.py`
> (see [Usage](#usage)).

---

## Table of contents

- [Why this project](#why-this-project)
- [Hardware & software stack](#hardware--software-stack)
- [Approach](#approach)
- [Fidelity to the real robot (sim → real)](#fidelity-to-the-real-robot-sim--real)
- [The skills (video segments)](#the-skills-video-segments)
- [Step-by-step detail](#step-by-step-detail-for-newcomers)
- [Findings](#findings)
- [Repository structure](#repository-structure)
- [Installing the simulation](#installing-the-simulation-beginner-guide--1-command)
- [Usage](#usage)
- [Exploration paths](#exploration-paths-beyond-hand-shaped-reward-ppo)
- [License](#license)

---

## Why this project

To show, in a way a non-specialist audience can read, **what RL on real hardware
concretely looks like**:

- we start from **zero** (random weights) — the robot can do nothing;
- we train **4096 robots in parallel** in the simulator;
- we **save regular checkpoints** so the progression can be replayed;
- we **render a video** where you see, side by side, the behaviour improving
  **and** the learning metric progressing.

---

## Hardware & software stack

| Item | Detail |
|---|---|
| GPU | AMD Radeon **RX 9070** (**ROCm** backend, `gs.amdgpu`) |
| Simulator | **Genesis** 1.2.x |
| RL algorithm | **PPO** via [`rsl-rl-lib`](https://github.com/leggedrobotics/rsl_rl) (≥ 5.0) |
| Parallel robots | **4096** environments |
| Robot | Unitree **Go2**, bare URDF `urdf/go2/urdf/go2.urdf` (12 DOF, no arm, no camera) |
| Rendering / training | training on **GPU**; video rendering on **CPU** — never both at the same time (see note below) |

---

## Approach

1. **Train one skill** — the **"walk" pilot from scratch**, then the following skills
   **warm-started** from it (to save time) — saving a checkpoint every 25 iterations
   (`save_interval`).
2. **Extract the learning metric** from the TensorBoard logs
   (`Train/mean_reward` = mean reward per episode — the most telling indicator:
   it **goes up** when the robot learns; `Train/mean_episode_length` = time before
   falling, very intuitive in locomotion).
3. **Render a composite video**: on the left the robot **at the current checkpoint**
   (struggling → staggering → walking), on the right **the curve filling in** at the
   same pace, and in a banner **the time equivalence**.

### Why the reward curve, and not "the loss"?

In reinforcement learning (PPO) the *loss* is not monotonic and does not tell you
"it is learning". The **rising mean cumulative reward** is the canonical, directly
readable indicator.

### "GPU time ↔ robot experience" equivalence

Each iteration makes 4096 robots live for 24 control steps at 50 Hz:

```
experience per iteration = 4096 envs × 24 steps × 0.02 s ≈ 1,966 s ≈ 32.8 min of robot experience
```

That is, for the 2500 iterations of walking, **≈ 57 days** of continuous experience of a
real robot, compressed into ~50 minutes of compute on the RX 9070. The video shows this
ratio live (elapsed GPU time ↔ equivalent months/years).

> ⚠️ **Practical note (AMD/ROCm)**: never launch a Genesis CPU render/eval while a GPU
> training is running — serialize the two (here: a GPU crash was observed otherwise).

---

## Fidelity to the real robot (sim → real)

The goal is not a "toy" robot: every physical constraint is calibrated on the **real
Unitree Go2**, so that a policy trained here means something on the hardware.

### Manufacturer specs (Unitree GO-M8010-6 motor)

| Quantity | Hip / Thigh | Knee |
|---|---|---|
| Nominal torque | **23.7 N·m** | **35.55 N·m** |
| Max speed | **30.1 rad/s** | **20.07 rad/s** |

Announced peak torque ~45 N·m (largest joint) — **it is the nominal 23.7/35.55 N·m that
is actually enforced during training**, not the peak. Joint ranges: hip ±60°,
thigh −90…200° (front) / −30…260° (rear), knee −156…−48°. *(Values from the official URDF + Unitree datasheets.)*

### What the simulation enforces

- **Torque bounded** to these values (`enforce_motor_limits`): the controller cannot
  request a torque that a real servo would not deliver.
- **Joint velocities** kept within spec via a penalty (`dof_vel_limit`).
- **Position targets clipped to the URDF joint ranges** (`clamp_targets_to_limits`,
  audit of 21/07): the real Go2's firmware rejects any target outside the joint
  range — the sim reproduces this clipping by construction. Without it, the network can
  "lean on" unreachable targets (artificially constant end-stop torque), a behaviour
  that does not transfer to hardware.
- **PD gains** kp = 20 / kd = 0.5 (standard Go2 locomotion values).

### Domain randomization (to transfer to the real world)

The robot learns **under perturbations**, not in a perfect world:

- **Ground friction** drawn randomly in **0.4–1.6** each episode (wet tiles →
  grippy concrete);
- **Random horizontal pushes** (~every 4 s, up to 1 m/s, with a phase specific to each
  robot) — shoves;
- **Randomized back payload** of **0–0.5 kg** (onboard accessories);
- **Sensor noise**: observations pass through IMU/encoder-style noise
  (gyro ±0.2 rad/s, attitude ±0.05, positions ±0.01 rad, velocities ±1.5 rad/s) — the
  real robot never "feels" its state perfectly.

These constraints are centralized in **`realism.py`** and injected by
`apply_realism(env_cfg, reward_cfg)` — called in `get_cfgs()`, so **every** training in
the project inherits them **by construction** (not merely by convention).
Any new skill must call `apply_realism`.

Spec sources: [Unitree Go2](https://www.unitree.com/go2/) ·
[motor datasheet / teardown](https://www.simplexitypd.com/blog/unitree-go2-motor-teardown/) ·
[QUADRUPED Docs](https://www.docs.quadruped.de/projects/go2/html/Overview_1.html).

---

## The skills (video segments)

| # | Skill | Base | Status |
|---|---|---|---|
| 1 | 🚶 Walk | `go2_train.py` (from scratch) | ✅ trained + rendered |
| 2 | 🏃 Run | warm-start from walking, speed curriculum | ✅ **2.7 m/s stable** (5 m/s sprint abandoned: stability > speed) |
| 3 | 🛌 Lie down | `go2_liedown_train.py` | ✅ trained + rendered |
| 4 | 🧎 Get up | `go2_getup_env.py` | ✅ trained + rendered (anti-bounce `vertical_settle`, see *Findings*) |
| 5 | 🐕‍🦺 Sit | `go2_sit_env.py` (projected gravity) | ✅ trained — *coarse+fine* pitch + tucked feet (`front_tuck`/`rear_tuck`) + gradual descent (v6), see *Findings* |
| 6 | 📡 LiDAR obstacle avoidance (visible rays) | hierarchical navigation over frozen locomotion | ✅ retrained with a richer setup (scan noise, narrow gap, unpredictable pedestrians) — 3 scenes rendered + full-crossing finale |
| 7 | 🧗 Cross obstacles | **7 separate terrain families** | ✅ 7 families trained + rendered (stairs, pyramid, slope, rough, curbs, gaps, waves) |

**Chosen approach**: first **one complete pilot** (the "walk" skill end to end —
training + curve + time overlay) to validate the visual style, then roll out the other
skills.

### Video files (real-time segments)

> The video renders (`videos/`) and the checkpoints (`logs/`) are **not versioned**
> (large, git-ignored); only the **GIF previews** ([`gifs/`](gifs/)) are in the
> repository. Each segment below is recreated by **training then rendering** with
> `pilot_walk_video.py` / `lidar_render.py` (see [Usage](#usage)):

| Skill | Rendered by | Generated file name |
|---|---|---|
| 🚶 Walk | `pilot_walk_video.py -e go2-walking` | `segment_go2-walking.mp4` |
| 🏃 Run | `pilot_walk_video.py -e go2-run` | `segment_go2-run.mp4` |
| 🛌 Lie down | `pilot_walk_video.py -e go2-liedown` | `segment_go2-liedown.mp4` |
| 🧎 Get up | `pilot_walk_video.py -e go2-getup` | `segment_go2-getup.mp4` |
| 🐕‍🦺 Sit | `pilot_walk_video.py -e go2-sit` | `segment_go2-sit.mp4` |
| 📡 LiDAR avoidance | `lidar_render.py --scenario fixes\|pietons\|goulot\|complet` | `segment_go2-lidar_*.mp4` |
| 🧗 Cross obstacles | `pilot_walk_video.py -e go2-terrain-<family>` | `segment_go2-terrain-*.mp4` |

---

## Step-by-step detail (for newcomers)

> **How to read a "training"** — We do NOT program the robot step by step. We give it a
> **reward** (a score) describing *what we want*, and the algorithm (PPO) adjusts on its
> own, by trial and error on **4096 robots in parallel**, the way of moving that
> maximizes that score. Below, each step = a different score.

### 1. 🚶 Walk

![The Go2 learns to walk](gifs/marche.gif)

2500 iterations, ~0.8 h of RX 9070 compute
(≈ 57 days of robot experience), reward climbing from −0.9 to ~8, episodes lasting
~870/1000 steps (the robot almost never falls any more).

**What it learns**: to start from random motion (it collapses) and discover a stable,
**upright** gait in the requested direction; then to **stop dead** when nothing is
asked of it (legs frozen, upright posture).

**How we train it** — we add up simple rewards:
- **move at the right speed** (main reward);
- **stay upright**: trunk level (`orientation`), body high on its legs
  (`base_height`, target 0.38 m), legs vertical underneath it (`hip_straight`);
- **stop cleanly**: on a zero command, base motionless (`stand_still`) **and**
  legs no longer moving (`dof_freeze_stop`) — 15 % of trials are run "at a standstill"
  so it also learns to *do nothing*;
- **soft penalties** against jerky motions and vertical hops.

This is the only step trained **from scratch**; the following ones often start from it
(*warm-start*) to save time.

### 2. 🏃 Run

![The Go2 learns to run](gifs/course.gif)

**Learns** (2.7 m/s stable): to accelerate well beyond walking while staying stable.

**Training**: we start from the walker and **progressively raise the commanded speed**
(curriculum); at high speed we **relax** the upright-posture constraints (a gallop has
to lower itself and engage the hips), but we **keep the stability** (level trunk, don't
fall) and the controlled stop.

**Speed choice: 2.7 m/s (stability > speed).** We tried pushing toward the **~5 m/s**
announced by Unitree (lab test of the Go2 EDU): under our bounded motor torques,
stability collapses (episode length 940 → 250, frequent falls).
So we keep **2.7 m/s**, the highest speed at which the robot runs fast **and**
holds up (~940 steps without falling). The lab spec is a theoretical ceiling, not a
training target when the priority is not falling (see *Findings*).

**Controlled stop after the sprint (dedicated finetune).** Cutting from 2.7 m/s → 0 at
once made the robot do a **front flip** (inertia tips it over): the gallop→standstill
*transition* is too rare during running training to be mastered. We add a finetune phase
where **30 %** of trials receive the zero command (instead of 12 %), to specifically
build stable deceleration (see *Findings*, "train the transition, not only the regime").

### 3. 🛌 Lie down

![The Go2 learns to lie down](gifs/coucher.gif)

**Learns**: to lower the body to the ground **slowly and in a controlled way**, then
stay there stably. **Training**: low height target (0.12 m) + **strong penalty on the
body's vertical velocity** — the target pulls downwards, the velocity brake enforces a
gradual, firmware-like descent. Two versions failed before this one: the low target
alone produced an **instantaneous** lie-down (a flop), and a temporal **trajectory**
target (scripted stepwise descent) was **not learnable** — PPO stayed standing, stuck in
the local optimum (see *Findings*, "put the movement in the reward").

### 4. 🧎 Get up

![The Go2 learns to get up](gifs/relever.gif)

**Learns**: from a lying/fallen position (random orientation: side, back…), to get back
on its feet **cleanly**. **Training**: strong reward for righting itself (projected
gravity) and regaining standing height; stillness is required **only once standing**
(gated on measured state, not on time) — during the rise the robot needs broad, fast
motions, and restricting them prevents the lift. A **soft** penalty on joint velocities
avoids "leg thrashing" without forbidding the push.

### 5. 🐕‍🦺 Sit

![The Go2 learns to sit](gifs/assis.gif)

**Learns**: rear on the ground, front raised (~40°, "sitting" posture), reached
**slowly**: static targets for pitch (via projected gravity) and hindquarters height +
a **vertical-velocity penalty** that forbids snapping into the posture at once (same
recipe as the lie-down, after the same instantaneous/trajectory double failure). No
`orientation` term: it would force the trunk level — the exact opposite of sitting.
**Third fix (23/07)**: widening the pitch-up Gaussian had unblocked the start (it no
longer stayed standing) but produced a **plateau at half-pitch** (front stays low) — the
half-sit was already "rewarding enough". Switched to a **coarse+fine** Gaussian (wide to
initiate, narrow to strongly reward completion) + raised weight, to force the **complete**
pitch-up (see *Findings*, "an overly wide Gaussian plateaus at mid-gesture").
**Fourth fix (23/07, v3 then v4)**: the *coarse+fine* did raise the chest
(nose-up + rear-down, reward ~48) but the robot settled on **spread stances** →
a **leaning/slumped** sit. Cause: *no term controlled foot position*, only pitch and
hindquarters height were targeted. A first **`front_tuck`** term (**front** feet tucked
under the shoulders) changed **almost nothing** (v3 ≈ v2): zooming into the render, the
real defect was **at the back** — the **rear legs** extend forward instead of folding
under the rump. Fix v4: a **`rear_tuck`** term targeting the **rear** feet (foot ↔ hip
attachment distance, since distance to the centre does not discriminate at the rear —
see *Findings*, "check which body part is failing" and "the right metric depends on the
segment"). All of this **without** relaxing any physics — Unitree torques/velocities,
URDF joint ranges, self-collisions and contacts stay enforced.
**Fifth fix (23/07, v5)**: the descent was still a bit **too instantaneous**. We
**strengthen the vertical-velocity penalty** (`lin_vel_z` −10 → −16, stronger than the
lie-down's −12 because the sit target is higher and therefore reached faster): the robot
**lowers itself more gradually** into the posture without snapping. The final posture
(rear folded, chest high) is **unchanged** — only the descent speed is. Same lever as the
lie-down, still *reward shaping* (we do not restrict the physics, we penalize velocity to
make the gesture slow and readable).
**Sixth fix (v6)**: on review, v5 (−16) was **still judged too fast**. We **double the
brake** (`lin_vel_z` −16 → **−32**): the descent now spreads over ~1.8 s
(standing → gradual lowering → sitting), without snapping. Important point: the
strengthened brake **did not restrict learning** — the reward still converges to **~117**
(identical to v5), and the final posture (rear on the ground, chest raised) is
**unchanged**. Confirmation that the "penalized vertical velocity" lever tunes the *speed*
of the gesture without compromising its *completion*.

> **Why the pitch target goes through projected gravity, not Euler angles** — audit of
> 21/07: in the Genesis convention (`quat_to_xyz`), a raised nose gives a **negative**
> pitch (verified empirically: a +20° rotation about +Y — nose down — returns +20°).
> The first Euler targets of "+40°/+80°" would therefore have trained the robot to
> **nose-dive** instead of pitching up. The reward now targets **projected gravity** in
> the body frame: nose raised by θ ⇔ `projected_gravity_x = −sin(θ)`. A formulation
> independent of any sign convention — the bug is impossible by construction — and with
> no gimbal singularity near 90°.

### 6. 📡 LiDAR obstacle avoidance *(visible rays)*

Three scenes, same policy (only the set dressing changes): fixed obstacles → moving
pedestrians → narrow gap.

![LiDAR — fixed obstacles, rays coloured by distance](gifs/lidar_fixes.gif)
![LiDAR — unpredictable moving pedestrians](gifs/lidar_pietons.gif)
![LiDAR — the gap in the barrier to cross](gifs/lidar_goulot.gif)

**Learns**: to cross a **walled corridor** of obstacles to a goal without collision:
8 cylinders including **3 unpredictable pedestrians** (heading, frequency and amplitude
drawn each episode), plus a **barrier across the corridor pierced by a gap**
(~1.3–1.9 m) at a random position — the robot has to find and cross the gap. The
corridor is **closed by two walls of posts** along its whole length and **leaving it
ends the mission with the same penalty as a collision**: a first version without walls
had learned to *go around the entire course from the outside* — perfectly rational
reward optimization, but zero avoidance (see *Findings*, "going around is a reward
hack"). To compensate, the internal passages are sized to remain crossable (obstacles
re-centred, ≥ ~0.85 m between an obstacle and the wall — the Go2's body is ~30 cm wide).
The **pedestrians move at walking pace** (peak speed ≈ 2π·f·A bounded to ~0.4–1.4 m/s)
rather than sprinting: a first setting let the frequency rise to pedestrians at
**2.6 m/s** (see *Findings*, "a pedestrian's speed is 2π·f·A").
A **two-layer** architecture:

- **Layer B (learned)**: a **navigation** policy reads the LiDAR scan
  (72 sectors) at 10 Hz and drives, in velocities (vx, vy, yaw), the walker's **frozen
  locomotion** policy at 50 Hz. The scan is **noised like a real sensor** (range noise
  ~3 cm + randomly lost echoes) — project rule: realism is not negotiable, even when it
  makes learning harder. A v1 without noise or barrier had validated the architecture
  (84 % success in ~400 iterations) before this full retraining.
- **Layer A (deterministic, NO AI)**: reflex emergency stop — minimum frontal distance
  from the scan, speed-adaptive threshold (`0.12 + 0.30·v`), progressive slowdown then a
  ban on moving forward. A "never any contact" guarantee cannot come out of a network:
  layer B proposes, layer A disposes. Inactive during training (the policy must learn
  avoidance without a crutch), active at inference and rendering — as at deployment.

The video **shows the rays** coloured by distance (green = clear, orange,
red = danger), and the segment is cut into **readable mini-scenes** (as for the
obstacles): first **fixed obstacles only**, then **moving pedestrians**, then
the **gap in the barrier** — same policy in all three, only the set dressing changes
(`lidar_render.py --scenario fixes|pietons|goulot|complet`). An "everything at once"
course would be unreadable on video. The **legend** recalls on screen the ray colour
code **and** the sensor noise ("realistic sensor noise: ~3 cm + lost echoes") — we show
that the input is degraded, not idealized.

**"Full crossing + controlled stop" finale (23/07).** The mini-scenes, timed to the
duration of one checkpoint, were **cutting off mid-crossing** (a framing flaw, not a
policy one). Fix: after the checkpoint loop, we render a **finale** that goes **all the
way to the goal** and then **comes to a stop** (zero locomotion command held for ~1.8 s,
**without reset** — the robot does not head back). Since a successful episode would
normally teleport the robot to the start, the finale first runs a **seed search**
(dry-run over ~40 seeds) to find a crossing that succeeds, then **replays** it while
burning in the LiDAR scan frame by frame (`_last_scan`) — a **deterministic** render,
with no re-simulation randomness. This is framing/rendering: the policy and the physics
are unchanged.

**Reading the success curve (two points annotated on screen).**
1. The **vertical dip at ~iteration 1499** is **not** a policy collapse but a **training-resume
   artefact**: on restart (`resume_from`), the **episode buffer is empty**, so the logged
   success rate drops to **0** for 1–2 iterations then **recovers immediately** (≈0.37 before,
   ≈0.37 after) — with no real impact on learning.
   *(Classic TensorBoard logging trap on a resumed run, see [Findings](#findings).)*
2. The rate **plateaus at ~41 %** in this **hard** scenario (sensor noise + unpredictable
   pedestrians + narrow gap): that is **honestly sub-optimal**. Raising it would go through
   **more iterations**, a difficulty **curriculum** and **automatic domain randomization (ADR)** —
   see [Exploration paths](#exploration-paths-beyond-hand-shaped-reward-ppo).

### 7. 🧗 Cross obstacles

One terrain family = one training run, one GIF:

![Straight staircase (5 cm steps)](gifs/terrain_escalier.gif)
![Stepped pyramid](gifs/terrain_pyramide.gif)
![Slope (~11°)](gifs/terrain_pente.gif)
![Rough ground (±10 cm)](gifs/terrain_accidente.gif)
![Curbs / discrete obstacles](gifs/terrain_rebords.gif)
![Gaps / separated slabs](gifs/terrain_trous.gif)
![Waves (the ground "rolls")](gifs/terrain_vagues.gif)

**Learns**: to walk on uneven ground. **One training run per obstacle family**, each with
its video segment: **slope** (~11°), **straight staircase** (5 cm steps),
**stepped pyramid** (5 cm), **rough ground** (±10 cm, gravel/rubble style),
**curbs / discrete obstacles** (up to 10 cm, kerb style), **gaps** (slabs separated by
10 cm voids, 25 cm deep — potholes/gratings) and **waves** (gentle 12 cm swell — the
ground "rolls"). **Water** is deliberately excluded: the solver used is rigid (realistic
fluid coupling is another simulator), and swimming is not a capability of the real Go2 —
simulating it would be showmanship, not sim→real. Why separate? Mixed terrain makes the
robot fail "at random" depending on the tile it spawns on — the video no longer tells a
progression but a lottery. Per family, each curve tells the learning *of that specific
obstacle*. We train in **all directions** (forward, backward, sideways, turning) up to
1.5 m/s + 15 % at a standstill: crossing while walking, not running (running on steps =
falls, the opposite of the stability priority). We reward **lifting the legs decisively**
(`feet_air_time`), and above all **staying stable** (level trunk) rather than "upright and
high" — on uneven ground the priority is not to topple — as well as **stopping dead**
even on a slope (controlled stop, legs frozen).

> **Four settings learned the hard way (22/07)**: (1) the slope was initially at
> **17°**, too steep for a walker coming from flat ground → brought back to **11°** (see
> *Findings*, "flat → uneven warm-start"); (2) an unbounded vertical-velocity penalty
> made training **diverge** as soon as a robot tumbled on the relief — it had to be
> bounded (see *Findings*, "unbounded penalty"); (3) with the standard 0.25 m heightfield
> mesh, a "staircase" of 7 cm steps is in reality **a ramp** — the riser is interpolated
> over 25 cm of run. Mesh brought down to **0.05 m** for staircase and pyramid: near-vertical
> risers, real geometry, real difficulty (see *Findings*, "a heightfield only makes steps
> at a fine mesh"); (4) on those real steps, the **"level trunk"** penalty prevented the
> robot from pitching up to climb → it fell; for staircase/pyramid it is **relaxed** and
> leg lift strengthened, steps lowered to **5 cm** (see *Findings*, "level-trunk stability
> prevents climbing").
> Each family follows the **peak-then-degradation** rule: we render the peak checkpoint
> (e.g. slope: `model_2550`, episode ~850), not the last one.

---

## Findings

Field notes, specialist level — what the training runs taught us:

### General (cross-cutting)

- **Sensor noise vs posture (the convergence lever).** Combining full *legged_gym*-level
  sensor noise (gyro 0.2; encoder-velocity 1.5 rad/s) with **aggressive** posture rewards
  (height `base_height` −150) **prevents from-scratch convergence**: plateau at reward ~5,
  episodes ~560/1000 (falls ~44 %). Rule adopted: **we never reduce the noise** (it is part
  of realism) — we **relax the upright-posture requirement**. The walker pilot converged
  (reward ~8, episodes ~870) with substantial noise and a measured posture.
- **"Robustifying" a firm-posture walker at full noise degrades it.**
  Resuming the walker at full noise *while keeping* the firm posture makes episodes
  **drop** (850 → 519 in 500 iterations). Conclusion: no need for a dedicated
  robustification step — the **running** and **terrain** segments (full noise + posture
  relaxed *by construction*) already robustify against noise **while learning their task**.
- **Angle conventions: verify empirically, never assume.** Audit of
  21/07: in Genesis, a raised nose = **negative** pitch — the positive Euler targets of
  the postures (sit +40°) were inverted and would have trained a "nose dive". Structural
  fix: pitch rewards target **projected gravity** (`pg_x = −sin θ`), insensitive to the
  convention and free of gimbal lock near 90°. Cost of the test that avoided the error:
  30 s of CPU; cost of the avoided error: ~1.5 h of GPU across two training runs.
- **Spec audit against the official URDF (21/07).** The limits in the manufacturer's file
  (efforts 23.7/23.7/35.55 N·m; velocities 30.1/30.1/20.07 rad/s) match exactly the
  constants in `realism.py`, in the right joint order. Added on that occasion the
  **clipping of targets to the URDF joint ranges** (firmware behaviour) — opt-in, so
  already-trained policies render identically.
- **⚠️ An UNBOUNDED penalty makes PPO diverge (the costliest trap, 22/07).**
  The terrain run warm-started from the walker (trained on flat ground) made **some**
  robots tumble on the relief → their vertical velocity rose to several m/s → the
  `lin_vel_z = v_z²` penalty (**unbounded**) exploded (mean reward spike at **−119**) →
  a catastrophic PPO gradient step → the policy collapsed (episode frozen at ~14 steps,
  robot lying down) and **never recovered** over hundreds of iterations. Identical, more
  discreet symptom on the staircase (spikes −24/−44). **General lesson**: any reward term
  that can physically explode (squared velocity, squared height error, contact)
  **must be bounded** (`.clip()` + `nan_to_num`) — otherwise a rare transient state is
  enough to destroy the training. The fix (`clip(max=4.0)`) is invisible in normal
  locomotion (v_z stays small there) and **changes nothing about realism**: it is
  *reward shaping*, not physics.
- **Over-training = peak then collapse (a quantified terrain example).** On the corrected
  slope: episode 14 → **~850 at the peak (it~2550)** → then degradation → **13** (it3177)
  → stagnates low (~70–90 at the end of the run). PPO does not stop on its own and drifts
  toward a more aggressive, brittle regime after the peak. **We select the checkpoint AT
  THE PEAK** (`model_2550` here), never the final one. Practical corollary: setting
  `max_iterations` generously and *choosing* afterwards beats cutting too early — each
  checkpoint is an independent frozen policy, replaying it reproduces its behaviour exactly
  (the peak is not "missed" because what followed degraded).
- **⚠️ A flaw in the BASE policy propagates through the whole chain (22/07, the
  sneakiest one).** The pilot walker — the base of *all* warm-starts AND the frozen
  locomotion of the LiDAR stack — had been trained **without self-collision**: its legs
  could pass through each other, and its "crossing" habits were transmitted to the child
  policies even when *they* were trained with collision active (odd running gait, legs
  brushing/passing through each other in the LiDAR render). Two lessons:
  (1) audit the **root policy** of a warm-start chain first — a flaw there costs you the
  whole lineage; (2) check the config **actually saved** (`cfgs.pkl`), not the current
  script — today's script does not tell you what ran yesterday (and a verification tool
  must be tested itself: our first pass read the wrong key and wrongly concluded that
  *everything* was collision-free).
- **The peak can fall BETWEEN two checkpoints (22/07).** On a warm-started run, the
  peak-then-collapse can be **fast**: the staircase peaks around iteration 45 then
  collapses in less than 60 iterations. With a save every 100, the true summit is
  **never recorded** — you only have already-degraded checkpoints.
  Fix: **save every 25** on *all* training runs (negligible disk cost,
  ~4.5 MB/checkpoint). Corollary: for a skill that peaks early, do a
  **short run** (250 iters) rather than a long one that will only diverge after the peak.
- **Resuming a run creates a new events file — merge it (22/07).** The automatic
  "best checkpoint" selector only read the **last** TensorBoard file; but a resume
  (`resume_from`) opens a new one, without the history. Result: it believed the peak was
  at the end of the finetune (an over-trained checkpoint) when the real peak was in the
  previous file. You must **merge all the `events*`** in the folder before looking for the
  argmax. Classic trap: a measuring tool that, in a particular case (a resume), measures
  the wrong thing with no visible error.
- **SAVE granularity ≠ DISPLAY density (video method).** Saving finely
  (every 25, for the peak) must not bloat the video. The render samples a fixed number of
  checkpoints along **front-loaded fractions** of the range (dense at the start where
  learning happens, spaced out afterwards), independent of `save_interval` and of the run
  length. Two distinct concerns — *not missing the peak* (saving) and *telling the learning
  story readably* (display) — cleanly decoupled.

### 🏃 Run

- **Running = stability, not upright posture.** Requiring a high, upright stance during
  the gallop is counter-productive (a gallop has to lower itself): PPO then "refuses to
  run". We keep the **stability** (level trunk, don't fall) and relax the posture. Same
  principle on terrain.
- **Speed is bounded by *physics*, not by an ideal.** With torques and joint velocities
  bounded to Unitree specs, the running reward becomes volatile near max speed: the robot
  converges to "as fast as the real motors allow" — exactly the behaviour we want to show.
- **Running speed: stability comes first, we do not force the lab spec.** Aiming for the
  announced ~5 m/s (Unitree lab test) under our bounded torques makes stability
  **collapse** (episode 940 → 250, frequent falls). We keep **2.7 m/s**: the speed at which
  the robot runs fast AND holds up (~940 steps without falling). The manufacturer spec is a
  theoretical ceiling, not a training target when the constraint is stability.
- **Train the TRANSITION, not only the regime (22/07).** A policy can excel in steady
  state (running at 2.7 m/s) and **fail at the transition** to another regime (stopping)
  if that transition is under-represented in training: an abrupt cut → a flip. The target
  regime is not enough; you have to expose the policy to the **switchovers** themselves
  (here: an increased fraction of zero-command trials occurring *during* the sprint).

### 🛌🧎🐕‍🦺 Postures (lie down / get up / sit)

- **Put the movement in the reward: the brake beats the choreography (22/07,
  two successive failures).** Rewarding "being lain down" (a static target) makes the
  robot dive **all at once** — the policy maximizes the fraction of the episode spent in
  the final posture. First attempted fix: turn the target into a firmware-style
  **temporal trajectory** (scripted stepwise descent, being early = penalized like being
  late). Result: **not learnable** — at the start, "standing" exactly matches the
  beginning of the trajectory, so no gradient ever initiates the descent, and PPO stays
  stuck standing (besides, the 45-D observation has no clock: following a stopwatch you
  cannot perceive is structurally fragile). The solution adopted, simpler and more robust:
  a **static target** (which does go down for sure) **+ a strong penalty on the body's
  vertical velocity** — the *where to go* stays trivial to learn, the *how* (slowly) is
  enforced by a brake, not by a choreography. Nuance for getting up: the final stillness
  is gated on the **measured state** (standing), not on time — you cannot know in advance
  how long a rise from the back takes.
- **The sit that never sits = a reward without gradient, NOT a manufacturer limit
  (22/07).** The robot stayed **standing** the whole time. Tempting false lead: "maybe
  it's blocked by the Go2's joint limits". Refuted by the video itself: if a mechanical
  end stop were blocking, the robot would **try** and freeze in a **partial** sit (targets
  clamped by the firmware); but it stays standing **without ever initiating the gesture**.
  It doesn't try → nothing prompts it to. Real cause: `sit_pitch` rewarded the pitch-up
  with a **too narrow** Gaussian (σ=0.25) on projected gravity — from standing
  (error ≈0.64) it is worth ~0.001, **flat**: pitching up a little earns almost nothing,
  hence **no gradient to START** (same trap as the phased lie-down, see above).
  And `rear_low` (−30) was ~2.7× weaker than the lie-down's height penalty (−80) → it did
  not lower enough. Fix: widen the Gaussian (σ=0.6 → a gradient from the very first degree
  of pitch-up) and strengthen the lowering (−55), **without touching the physics or the
  limits**: `apply_realism` stays active (torque 23.7/35.55 N·m,
  velocity 30.1/20.07 rad/s, `clamp_targets_to_limits`). **Non-negotiable principle**: a
  posture that fails is fixed by **reward shaping** (creating a gradient), NEVER by
  loosening a physical or manufacturer constraint — a reward cannot bypass the limits
  anyway (it makes the gesture *wanted*; the hard floor remains). The real Go2 sits within
  its limits, so the gesture is honestly reachable.
- **An overly WIDE Gaussian plateaus at mid-gesture (sit, 23/07).** Widening the pitch-up
  Gaussian (σ 0.25 → 0.6) had fixed the non-start — but introduced the opposite defect: the
  robot **spread out at half-pitch** (front low) instead of sitting properly. Cause: a wide
  Gaussian makes the **half-gesture already very rewarding** — at ~20° of nose-up it is
  already worth ~0.78, and going all the way to 40° (a real sit) only adds ~0.2, not enough
  to justify the effort and the instability → a local optimum halfway. The width that
  *helps starting* **hurts completing**: a single parameter cannot tune both ends of the
  gesture. Fix: a **coarse+fine** reward — the sum of two Gaussians, a **wide** one
  (σ=0.6) keeping the initiation gradient from standing, and a **narrow** one (σ=0.18)
  strongly rewarding the *complete* gesture (partial→complete goes from ~0.35 to 1.0
  instead of +0.2) — plus a raised weight (6→9) to dominate. General lesson: when a single
  reward has to both **initiate** a gesture from far away **and** demand its **precise
  completion**, one scale is not enough; superimposing a coarse scale (exploration) and a
  fine scale (exploitation) decouples the two — without touching the physics (pure *reward
  shaping*, `apply_realism` intact).
- **A correct posture plateaus if nothing constrains the STANCE (sit v3, 23/07).** The
  *coarse+fine* handled pitch (nose-up) and hindquarters height well — reward ~48 — but
  the robot reached them by **spreading a pair of legs**: a leaning sit, not an upright one.
  Cause: the reward described *trunk orientation* and *height*, never **where the feet
  land** → the solution space included the slumped variant, equally rewarding. Lesson:
  specifying a posture by angles/heights alone under-determines the gesture; you also have
  to **constrain the contact points** (a bounded [0,1] term, same scale as `sit_pitch`,
  pure *reward shaping* — `get_links_pos()` **reads** the post-contact position, the
  physics does not move).
- **⚠️ Check WHICH body part is failing BEFORE shaping (sit v3→v4, 23/07 —
  lesson learned).** The first stance fix (`front_tuck`) targeted the **FRONT** feet →
  **almost no effect** (v3 ≈ v2). By **zooming** into a render frame, the real defect
  appeared: it was **not** the front (roughly correct) but the **REAR** — the rear legs
  **extend forward** instead of folding under the rump. I had shaped the **wrong pair of
  legs**. Lesson: an "eyeball" diagnosis on a wide shot is misleading; **identify the
  faulty segment precisely (zoom, front/rear identification) before spending a GPU cycle** —
  a perfectly written reward term on the wrong target fixes nothing. Fix v4 `rear_tuck` on
  the rear feet.
- **The right METRIC depends on the segment (sit v4, 23/07).** For the **front** feet,
  "tucked" = *small distance to the body CENTRE* (`front_tuck`). For the **rear** feet,
  that same metric **fails**: a rear foot spreading forward gets **closer** to the centre →
  distance to the centre would wrongly reward it. You have to measure the distance from the
  rear foot to **its hip attachment** (`*_thigh` link): folded = foot under the hip (small
  distance), extended forward = large distance. Lesson: a "tuck" metric is not transposable
  as-is from one limb to another — check that it **properly discriminates** the good and bad
  cases for THAT limb (here the sign of the defect flips between front and rear).
- **A "bounce" when getting up is not a violation of motor limits — it is CONTACT
  (22/07).** Video feedback: the robot seems to "bounce off the ground" while getting up.
  Yet the **actuator** limits (torque/velocity/position) are scrupulously enforced
  (`apply_realism` active in getup training) — they have nothing to do with a bounce.
  A bounce is a matter of **contact dynamics**: no restitution is configured
  (quasi-inelastic contact by default), but an overly **vigorous** rise produces
  **vertical-velocity spikes** that make the body slam/bounce. Fix *through the reward*,
  without touching the physics: a `vertical_settle` term (−0.6) penalizes the base's `v_z²`.
  The asymmetry is the key — a smooth rise has `v_z`≈0.3 m/s (`v_z²`≈0.1, near-zero cost)
  whereas a bounce has spikes >1 m/s (`v_z²`>1, high cost): the term targets abrupt impacts
  **without** braking the necessary rise. If a residual bounce persisted, it would be of
  **timestep** origin (substeps=2) and would be handled by refining the contact — never by
  loosening a manufacturer limit.

### 📡 LiDAR

- **Going around is an avoidance reward hack (22/07).** Without walls, the LiDAR policy
  learned to **go around the course from the outside**: maximal progress toward the goal,
  minimal proximity — optimal reward, zero avoidance. That is the *rational* behaviour for
  the reward given, not a learning bug. Structural fix: **close the world** (an
  impassable walled corridor, visible to the LiDAR) and make leaving it a penalized
  termination, while **widening the internal passages** so the honest strategy stays
  feasible. Generalizable: if a loophole exists, PPO will find it — removing it physically
  beats penalizing it finely.
- **A pedestrian's speed is 2π·f·A (22/07).** The pedestrians on the LiDAR course
  oscillate laterally (amplitude A, frequency f): their **peak** speed is
  `2π·f·A`, not `f` or `A` in isolation. A draw that looked "reasonable" term by term
  (f up to 0.35 Hz, A up to 1.2 m) actually produced pedestrians at **2.6 m/s**
  — a sprint, not a walk, unrealistic and unfairly hard to avoid. We now bound the
  **product** to stay at ~0.4–1.4 m/s (human pace). Lesson: bound the *physically
  observed* quantity (the speed), not its factors taken separately.

### 🧗 Crossing obstacles (terrain)

- **Flat → uneven warm-start: soften the entry step.** A 17° (30 %) slope is too steep
  right away for a walker that only knows flat ground: even without the divergence above,
  it has no basis for holding on. Brought back to **11°**, it adapts progressively.
  Generalizable: a terrain's difficulty must stay within reach of the starting skill,
  otherwise the warm-start is useless.
- **A heightfield only makes steps at a fine mesh (22/07).** The video review revealed
  that the "staircase" that had been trained was in fact **a ramp**: at a horizontal mesh
  of 0.25 m, a 7 cm riser is interpolated over 25 cm of run (~16° of local slope). The
  render AND the physics were wrong — the robot never encountered a vertical step. Mesh
  brought down to **0.05 m** (riser over 5 cm, ~54°) for staircase and pyramid: the
  geometry is verified **visually on a CPU image before any GPU** (30 s that save hours).
  General lesson: on a heightfield, any "vertical" edge has a real slope of
  `height/mesh` — check the image, not the parameters.
- **"Level trunk" stability prevents climbing a staircase (22/07).** The general rule for
  uneven ground — *reward a level trunk so as not to topple* — **backfires** on steps: to
  place a leg on the riser above, the robot has to **pitch the front up** (nose in the
  air). A strong orientation penalty (−3.5) fights exactly that gesture → the robot bumps
  into the staircase and falls. Step-specific fix: orientation **relaxed to −0.8**
  (free pitch), `feet_air_time` strengthened (decisive lift to land on the step),
  edge contacts tolerated, steps lowered from 7 → **5 cm**. Lesson: a posture constraint
  that is right "on average" can forbid *the* key gesture of a sub-case — audit per terrain
  type, not globally.
- **A RENDERING flaw is not a POLICY flaw — diagnose before retraining
  (22/07).** On the staircase, the robot did a "flip" when stopping. First
  reflex: finetune the stop. Wrong reflex. The real cause: the terrain is a grid of 9 m
  tiles where **each staircase rises independently** → there is a **cliff** (~1 m) at the
  junction between two tiles. At **render** time (a single environment), the respawn drew
  a **random** tile and position — sometimes near an edge — and three seconds of climbing
  made it **cross the cliff**: the robot walked into the void and toppled. Crucial point:
  the locomotion policy is **blind** (proprioception only, no terrain scan) — it *cannot*
  avoid an invisible edge, so **no training would fix this**. The fix is on the rendering
  side, at zero GPU cost: a **deterministic spawn at the centre of a tile**
  (`place_on_terrain_center` in `pilot_walk_video.py`), the climb staying far from the
  edges. The pyramid did not have the bug: its steps rise toward the **central apex**, so
  the robot climbs *away from* the edges. Lesson: faced with aberrant behaviour on video,
  distinguish *what the policy learned* from *what the staging imposes on it* — here, the
  user corrected an initial wrong diagnosis, and the real fix saved a pointless finetune.

---

## Repository structure

```
install.sh            # installs EVERYTHING (ROCm distrobox container + venv + Genesis) — 1 command
uninstall.sh          # cleanly removes container + venv (keeps code, logs, videos)
go2_train.py          # from-scratch PPO training (upright walking, bare robot)
go2_env.py            # Genesis environment (12 DOF, rewards + realistic motor/DR)
realism.py            # centralized sim->real REALISM (Unitree specs + randomization) — mandatory
go2_run_train.py      # RUN: speed curriculum → manufacturer limit (stability > posture)
go2_terrain_env.py    # ALL-TERRAIN env: heightfield, spawn on relief, relative base height
go2_terrain_train.py  # CROSS: 7 separate obstacle families (--family pente|escalier|…)
go2_lidar_env.py      # LIDAR: hierarchical nav (72-sector scan → vx,vy,yaw), walled corridor
go2_lidar_train.py    # LIDAR: navigation policy training
go2_getup_env.py      # GET UP: start lying at random orientation, stillness gated on state
go2_getup_train.py
go2_liedown_train.py  # LIE DOWN: low target + vertical-velocity brake (slow descent)
go2_sit_env.py        # SIT: pitch via projected gravity + velocity brake
go2_sit_train.py
pilot_walk_video.py   # composite video render: robot + learning curve + time overlay
lidar_render.py       # LiDAR render: debug rays coloured by distance
_peakcap.py           # PEAK checkpoint selector (merges the TensorBoard events)
LICENSE               # MIT license
gifs/                 # GIF previews (one per skill) — versioned
logs/                 # checkpoints + TensorBoard logs      (git-ignored)
videos/               # high-resolution renders             (git-ignored)
```

---

## Installing the simulation (beginner guide — 1 command)

Everything runs in a **[distrobox](https://github.com/89luca89/distrobox) container**:
a "Linux inside your Linux" that carries the whole GPU stack (ROCm + PyTorch already
compiled by AMD) **without modifying anything on your system**. You can remove it all
cleanly at the end. Tested on: **RX 9070**, image
`rocm/pytorch:rocm7.2.4` (Ubuntu 24.04, Python 3.12, PyTorch 2.10),
**Genesis 1.2.2**, **rsl-rl-lib 5.4.2**.

### Prerequisites (once)

A Linux with `distrobox` and `podman` (or docker):

```bash
# Fedora
sudo dnf install distrobox podman
# Debian / Ubuntu
sudo apt install distrobox podman
# Arch
sudo pacman -S distrobox podman
```

For an AMD GPU, your user must have access to the GPU:
`sudo usermod -aG video,render $USER` (then log out and back in).

### Automatic installation

```bash
git clone <repo-url> go2-rl && cd go2-rl
./install.sh
```

The script: creates the `genesis-box` container from the official AMD image (~15 GB
on first download — be patient), creates the `~/venvs/genesis` venv which **inherits
the image's ROCm PyTorch** (no compilation, no exotic wheels), installs
Genesis + the dependencies (pinned versions), then checks that the GPU is visible.

Variants: `BOX_NAME`, `BOX_IMAGE`, `VENV_DIR` as environment variables.
**NVIDIA GPU**: replace the image with a CUDA+PyTorch image and follow the same
steps (`BOX_IMAGE=... ./install.sh` — the script detects where the image's PyTorch
lives, `/opt/venv` or the system python, and the venv inherits it; untested on the
NVIDIA side, feedback welcome). **Without a GPU**: untested — the training scripts select
the GPU backend (only `go2_lidar_train.py` offers a `--cpu` flag, usable for small trials
with `-B 256`).

### Uninstallation

```bash
./uninstall.sh     # removes venv + container (+ image if you confirm)
```

Your code, `logs/` (checkpoints) and `videos/` (renders) are kept.

---

## Usage

Always **enter the container** first, then activate the venv:

```bash
distrobox enter genesis-box
source ~/venvs/genesis/bin/activate
cd <project-folder>

# 1) Train walking from scratch (checkpoints every 25 iters in logs/go2-walking)
python go2_train.py -e go2-walking -B 4096 --max_iterations 2500

# 2) Follow the learning live (another terminal, same container)
tensorboard --logdir logs

# 3) Render the composite video robot + curve + time overlay (AFTER training)
python pilot_walk_video.py -e go2-walking

# Other skills: go2_run_train.py (running), go2_terrain_train.py --family pente,
# go2_getup_train.py, go2_liedown_train.py, go2_sit_train.py,
# go2_lidar_train.py — then lidar_render.py for the render with rays.
```

> ⚠️ ROCm reminder: **never** render/eval with Genesis while a GPU training is
> running (crash observed) — always one AFTER the other.

---

## Exploration paths (beyond hand-shaped-reward PPO)

This project is **hand-shaped-reward PPO**, one *named* skill at a time, with maximal
realism. Several directions would extend it — ranked here by **alignment with the
project's goal** (sim→real, eventually a deployment on a physical Go2 EDU). The Genesis
ecosystem being young, most of these recipes were published on **Isaac Gym / Isaac Lab**
and need porting.

**Strongly aligned (generalization + sim→real):**

- **Automatic terrain curriculum.** Today one terrain family = one separate training run
  (an educational choice: one readable curve per obstacle). For a *generalist* crosser,
  take up the **legged_gym** scheme: the grid of sub-terrains (`gs.morphs.Terrain`,
  `n_subterrains`) becomes a **difficulty matrix** (row = level, column = type), each
  env carries a `terrain_level`, promoted when it crosses / demoted when it stalls. A
  detail that matters: at the max level, **send the env back to a random level** (otherwise
  the population piles up on the last level and diversity collapses). One notch above:
  **PLR / ACCEL** — sample and mutate the heightfields maximizing *regret* (TD error), for
  a difficulty generator **with no ceiling**.
- **Automatic domain randomization (ADR).** Our randomization uses *fixed* ranges
  (`realism.py`). ADR **widens each range** (friction, mass ±, CoM, PD gains, actuator
  delay, observation noise) as soon as performance exceeds a threshold — the domain opens
  as far as the policy can take it. **Known trap**: do NOT randomize from iteration 0
  (the base gait cannot emerge) — train 500–1000 clean iterations, *then* open up. This is
  the natural continuation of our sim→real requirement.

**Useful mainly for very long / open-ended runs:**

- **Loss of plasticity.** On a loop meant to run for days, a network *freezes* its ability
  to learn. Countermeasures to wire in early: **ReDo** (periodically reinitialize
  near-inactive neurons), **LayerNorm** on actor + critic, **L2 regularization toward the
  init weights** (not toward zero). Without this a long run plateaus and the curriculum
  gets blamed unfairly.

**Research pivot (departs from the "named skills" narrative):**

- **Unsupervised skill discovery.** Remove the task reward and let behaviours *emerge*:
  **DIAYN** (intrinsic reward `log q(z|s)`), **METRA** (clearly better than DIAYN in
  locomotion — DIAYN tends to find "16 ways to stand still"), or the simpler **RND** as an
  exploration bonus on top of the existing PPO (~30 lines). Scientifically interesting, but
  it is a *different* project from "showing a robot learn identifiable gestures" — to be
  considered a branch, not an evolution.

**Alternative starting bases.** Genesis's official Go2 example is **deliberately
minimalist** (flat ground, no curriculum, no DR, simplified rewards) — hence the fact that
here terrain, curriculum and randomization were **hand-built**. To avoid rewriting the
infrastructure, **`lupinjia/genesis_lr`** (a legged_gym → Genesis port: terrains, CTS
teacher-student, AMP, DeepMimic already wired in) is a heavier but complete base. In every
case: **pin the versions** (`genesis-world`, `rsl_rl` — the API moved between 0.2 and 1.0).

> **Opinion (for this project specifically).** The two highest-return building blocks are
> the **terrain curriculum** and **ADR**: they directly serve generalization and transfer
> to the real Go2 EDU, in the continuity of the sim→real philosophy. Skill discovery is a
> fine research topic but reorients the message. Practical advice: if you find yourself
> spending more than two days on Genesis *plumbing* rather than on the algorithm, prototype
> the curriculum logic on Isaac Lab (where it is provided), then port it over.

---

## License

**MIT** — see the [`LICENSE`](LICENSE) file. Free reuse, including commercially, provided
the license notice is kept.
