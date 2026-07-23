"""SE COUCHER — entraînement (robot nu). Le robot part DEBOUT et se couche de façon
CONTRÔLÉE et PROGRESSIVE (pas d'un coup), puis reste immobile. Warm-start marcheur.

Approche v3 (22/07, après échec de la trajectoire phasée) : la cible-trajectoire
temporelle était trop dure à apprendre — PPO restait coincé DEBOUT (au départ, debout
= conforme au début de la trajectoire → aucun gradient n'amorçait la descente). On
revient donc à une CIBLE STATIQUE BASSE (`base_height` 0,12 m — descend à coup sûr,
comportement prouvé) mais on LIMITE LA VITESSE VERTICALE du corps (`lin_vel_z` fort) :
la hauteur cible tire le corps vers le bas, la pénalité de vitesse verticale rend la
descente LENTE et graduelle (~firmware StandDown), sans exiger un suivi de trajectoire
fragile. `dof_freeze_stop`/`stand_still` figent le robot une fois couché.

    python go2_liedown_train.py -e go2-liedown -B 4096 --max_iterations 1200
"""
import argparse
import glob
import os
import pickle
import re
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env
from go2_train import get_train_cfg, get_cfgs
from realism import apply_realism

WARM_EXP = "go2-walking"
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def last_ckpt(exp):
    its = [int(m.group(1)) for f in glob.glob(f"logs/{exp}/model_*.pt")
           if (m := re.search(r"model_(\d+)\.pt$", f))]
    return max(its) if its else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", default="go2-liedown")
    p.add_argument("-B", "--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=1200)
    p.add_argument("--resume_from", type=int, default=0)
    args = p.parse_args()

    gs.init(backend=getattr(gs, "amdgpu", None) or gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["self_collision"] = True
    env_cfg["links_to_keep"] = FEET
    env_cfg["termination_if_pitch_greater_than"] = 1.0e4   # le corps descend : pas de mort
    env_cfg["termination_if_roll_greater_than"] = 1.0e4
    env_cfg["episode_length_s"] = 6.0
    env_cfg["zero_cmd_frac"] = 1.0                          # commande nulle : se coucher sur place

    command_cfg["lin_vel_x_range"] = [0.0, 0.0]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]

    reward_cfg["base_height_target"] = 0.12    # corps au sol (cible statique = descend à coup sûr)
    reward_cfg["reward_scales"] = {
        "base_height": -80.0,       # descendre le corps très bas (moteur principal)
        "lin_vel_z": -12.0,         # LIMITE la vitesse verticale → descente LENTE/graduelle (pas d'un coup)
        "orientation": -3.0,        # tronc à plat (ne se couche pas en tas / sur le flanc)
        "stand_still": -1.0,        # pas de dérive latérale/cap
        "dof_freeze_stop": -0.02,   # pattes figées une fois couché
        "action_rate": -0.005,
    }
    apply_realism(env_cfg, reward_cfg)

    train_cfg = get_train_cfg(args.exp_name)
    log_dir = f"logs/{args.exp_name}"
    resume = args.resume_from > 0
    if not resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    env = Go2Env(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                 reward_cfg=reward_cfg, command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if resume:
        runner.load(f"{log_dir}/model_{args.resume_from}.pt")
        print(f"REPRISE model_{args.resume_from}")
    else:
        warm = f"logs/{WARM_EXP}/model_{last_ckpt(WARM_EXP)}.pt" if last_ckpt(WARM_EXP) else ""
        if warm and os.path.exists(warm):
            runner.load(warm)
            print(f"warm start depuis {warm}")
        else:
            print("(from-scratch)")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
