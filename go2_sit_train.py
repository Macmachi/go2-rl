"""S'ASSEOIR — entraînement (robot nu). Départ debout → posture assise CONTRÔLÉE et
PROGRESSIVE (cibles statiques + vitesse verticale limitée → lente, pas d'un coup)
tenue immobile. Warm-start depuis le marcheur.

Pas de terme `orientation` (il forcerait le tronc à plat = l'inverse de l'assise).
La cible d'assiette passe par la GRAVITÉ PROJETÉE (signe résolu, audit 21/07) —
`sit_pitch_target` en degrés « nez relevé », sans ambiguïté de convention.
Réalisme hérité (couples/vitesses Unitree bornés + DR + bruit).

    python go2_sit_train.py -e go2-sit -B 4096 --max_iterations 1200
"""
import argparse
import glob
import os
import pickle
import re
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_sit_env import Go2SitEnv
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
    p.add_argument("-e", "--exp_name", default="go2-sit")
    p.add_argument("-B", "--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=1200)
    p.add_argument("--pitch", type=float, default=40.0, help="cible de tangage assis (degrés)")
    p.add_argument("--resume_from", type=int, default=0)
    args = p.parse_args()

    gs.init(backend=getattr(gs, "amdgpu", None) or gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["self_collision"] = True
    env_cfg["links_to_keep"] = FEET
    env_cfg["termination_if_pitch_greater_than"] = 1.0e4   # le tronc s'incline : pas de mort
    env_cfg["termination_if_roll_greater_than"] = 1.0e4
    env_cfg["episode_length_s"] = 6.0
    env_cfg["zero_cmd_frac"] = 1.0                          # s'asseoir sur place

    command_cfg["lin_vel_x_range"] = [0.0, 0.0]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]

    reward_cfg["sit_pitch_target"] = args.pitch
    reward_cfg["sit_height_target"] = 0.26
    # Assise CONTRÔLÉE (v3, 22/07) : cibles STATIQUES + limitation de vitesse verticale
    # (`lin_vel_z`) → l'assise est atteinte à coup sûr mais LENTEMENT (pas d'un coup).
    reward_cfg["reward_scales"] = {
        "sit_pitch": 9.0,           # assiette nez relevé — DOMINANT (6→9, correctif 23/07) + gaussienne coarse+fine (env) pour forcer le cabrage COMPLET, pas un demi-assis
        "front_tuck": 5.0,          # pieds avant rangés sous les épaules (correctif v3 23/07) : écarte l'assise ÉTALÉE vers l'avant de la v2 ; bounded [0,1] comme sit_pitch, ne le domine pas
        "rear_tuck": 7.0,           # pieds ARRIÈRE repliés sous la croupe (correctif v4 23/07) : le vrai défaut (v3≈v2) — l'arrière s'étalait ; distance pied↔cuisse, bounded [0,1], poids > front_tuck car c'est LE point à corriger
        "rear_low": -55.0,          # arrière-train à mi-hauteur : abaissement DÉCISIF (−30→−55 : à −30 trop faible, il restait debout ; < −80 du coucher pour s'asseoir sans se coucher)
        "lin_vel_z": -32.0,         # LIMITE la vitesse verticale → transition LENTE/progressive (v6 23/07 : −16→−32, v5 encore jugée « trop rapide » à la revue — on double le frein pour une descente franchement progressive ; surveiller que ça ne bride pas la convergence, sinon revenir vers −26)
        "stand_still": -1.0,        # base sans dérive latérale/cap
        "dof_freeze_stop": -0.03,   # pattes figées une fois assis
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

    env = Go2SitEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
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
