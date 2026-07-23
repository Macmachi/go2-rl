"""SE RELEVER — entraînement (robot nu). Le robot part couché (orientation aléatoire)
et apprend à se remettre debout puis tenir. Warm-start depuis le marcheur.

Points clés : AUCUNE terminaison sur tangage/roulis (il commence au sol), commande NULLE
(il se relève sur place), self-collision ON (démêler les pattes). Réalisme hérité.

    python go2_getup_train.py -e go2-getup -B 4096 --max_iterations 1500
"""
import argparse
import glob
import os
import pickle
import re
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_getup_env import Go2GetupEnv
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
    p.add_argument("-e", "--exp_name", default="go2-getup")
    p.add_argument("-B", "--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=1500)
    p.add_argument("--init_ckpt", default="")
    p.add_argument("--resume_from", type=int, default=0)
    args = p.parse_args()

    gs.init(backend=getattr(gs, "amdgpu", None) or gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["self_collision"] = True                     # démêler les pattes au sol
    env_cfg["links_to_keep"] = FEET
    env_cfg["termination_if_pitch_greater_than"] = 1.0e4  # il COMMENCE au sol : pas de mort
    env_cfg["termination_if_roll_greater_than"] = 1.0e4
    env_cfg["episode_length_s"] = 8.0                    # le temps de se relever et tenir
    env_cfg["zero_cmd_frac"] = 1.0                       # commande nulle : se relever SUR PLACE

    command_cfg["lin_vel_x_range"] = [0.0, 0.0]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]

    reward_cfg["base_height_target"] = 0.32
    # CORRECTION 22/07 : se relever EXIGE de grands mouvements rapides. Les pénalités
    # de mouvement (action_rate, stand_still, dof_freeze_stop, similar_to_default)
    # punissaient justement le geste nécessaire (action_rate explosait à −144, 100× les
    # autres → le robot apprenait à rester couché immobile). On COUPE ces pénalités
    # pendant la phase de lever et on RENFORCE les moteurs (se redresser + hauteur debout
    # + poser les pieds). L'immobilité finale reste encouragée mais faiblement.
    # v2 (22/07, rendu « catastrophe » : gesticulation) : l'immobilité est désormais
    # conditionnée à l'ÉTAT debout (`hold_upright` dans l'env) — zéro frein pendant la
    # remontée, arrêt NET une fois debout — plus une pénalité DOUCE des vitesses
    # articulaires (`dof_vel_soft`) pour préférer un relevé fluide au battage de pattes.
    reward_cfg["reward_scales"] = {
        "upright": 5.0,             # SE REDRESSER (gravité projetée z → -1) — moteur principal RENFORCÉ
        "base_height": -60.0,       # atteindre/tenir la hauteur debout (driver de la station)
        "feet_down": 1.5,           # poser les 4 pieds (renforcé)
        "hold_upright": -0.4,       # immobilité STRICTE dès que debout (gate sur l'état, pas le temps)
        "dof_vel_soft": -0.0008,    # anti-gesticulation doux (laisse la poussée du lever)
        "action_rate": -0.0003,     # anti-thrash minimal
        "vertical_settle": -0.6,    # ANTI-REBOND (retour vidéo 22/07) : lisse la remontée, pénalise les pics de v_z ; asymétrique (relevé fluide ~0 coût) — n'entrave pas la remontée
        # RETIRÉS : dof_freeze_stop et similar_to_default fightaient directement le geste de lever.
    }
    apply_realism(env_cfg, reward_cfg)                   # couples/vitesses bornés + DR + bruit

    train_cfg = get_train_cfg(args.exp_name)
    log_dir = f"logs/{args.exp_name}"
    resume = args.resume_from > 0
    if not resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    env = Go2GetupEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                      reward_cfg=reward_cfg, command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if resume:
        runner.load(f"{log_dir}/model_{args.resume_from}.pt")
        print(f"REPRISE model_{args.resume_from}")
    else:
        warm = args.init_ckpt or (f"logs/{WARM_EXP}/model_{last_ckpt(WARM_EXP)}.pt"
                                  if last_ckpt(WARM_EXP) else "")
        if warm and os.path.exists(warm):
            runner.load(warm)
            print(f"warm start depuis {warm}")
        else:
            print("(from-scratch)")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
