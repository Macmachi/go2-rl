"""SEGMENT LiDAR — entraînement de l'ÉVITEMENT appris (robot nu).

La policy de NAVIGATION apprend à piloter le marcheur GELÉ pour rejoindre un but en
contournant des obstacles FIXES + MOBILES, vue uniquement par son LiDAR 72 secteurs.
Locomotion gelée = le marcheur `go2-walking` (dernier checkpoint). Sol PLAT (on isole
l'évitement). Entraînement FROM-SCRATCH de la nav (tâche nouvelle, obs 80-D dédiée).

    python go2_lidar_train.py -e go2-lidar -B 4096 --max_iterations 1500
    python go2_lidar_train.py -e go2-lidar --cpu --num_envs 16 --max_iterations 2   # smoke test
"""
import argparse
import glob
import os
import pickle
import re
import shutil

import genesis as gs

from go2_lidar_env import Go2LidarAvoidEnv

LOCO_EXP = "go2-walking"        # locomotion gelée = le marcheur droit réaliste


def get_train_cfg(exp_name):
    # réglages nav (repris du système validé) : LR/entropy/std plus bas que la loco —
    # la policy nav (72 entrées de scan à interpréter) est plus délicate à stabiliser.
    return {
        "algorithm": {
            "class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01,
            "entropy_coef": 0.005, "gamma": 0.99, "lam": 0.95, "learning_rate": 3e-4,
            "max_grad_norm": 1.0, "num_learning_epochs": 5, "num_mini_batches": 4,
            "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5,
                                 "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 32, "save_interval": 25, "run_name": exp_name,   # fin (demande 22/07)
        "logger": "tensorboard",
    }


def _latest_ckpt(exp):
    its = [int(m.group(1)) for f in glob.glob(f"logs/{exp}/model_*.pt")
           if (m := re.search(r"model_(\d+)\.pt$", f))]
    return max(its) if its else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", default="go2-lidar")
    p.add_argument("-B", "--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=1500)
    p.add_argument("--loco_exp", default=LOCO_EXP)
    p.add_argument("--loco_ckpt", type=int, default=0, help="0 = dernier checkpoint dispo")
    p.add_argument("--init_ckpt", default="", help="warm start nav explicite (défaut : from-scratch)")
    p.add_argument("--resume_from", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="smoke test CPU (peu d'envs)")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    loco_ckpt = args.loco_ckpt or _latest_ckpt(args.loco_exp)
    if loco_ckpt is None:
        raise SystemExit(f"Pas de checkpoint locomotion dans logs/{args.loco_exp}/")
    loco_path = f"logs/{args.loco_exp}/model_{loco_ckpt}.pt"
    print(f"locomotion gelée : {loco_path}")

    with open(f"logs/{args.loco_exp}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, _rc, command_cfg, _tc = pickle.load(f)
    env_cfg = dict(env_cfg)
    env_cfg.pop("terrain_cfg", None)              # sol PLAT : on isole l'évitement
    env_cfg["base_init_pos"] = [0.0, 0.0, 0.42]   # départ à l'origine, face à +x
    reward_cfg = {"reward_scales": {}}            # rewards loco off ; la nav a les siennes

    if args.cpu:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning", seed=args.seed)
    else:
        gs.init(backend=getattr(gs, "amdgpu", None) or gs.gpu, precision="32",
                logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2LidarAvoidEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                           reward_cfg=reward_cfg, command_cfg=command_cfg,
                           locomotion_ckpt=loco_path)

    from rsl_rl.runners import OnPolicyRunner
    train_cfg = get_train_cfg(args.exp_name)
    log_dir = f"logs/{args.exp_name}"
    resume = args.resume_from > 0
    if not resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if resume:
        runner.load(f"{log_dir}/model_{args.resume_from}.pt")
        print(f"REPRISE model_{args.resume_from}")
    elif args.init_ckpt and os.path.exists(args.init_ckpt):
        runner.load(args.init_ckpt)
        print(f"warm start nav depuis {args.init_ckpt}")
    else:
        print("(nav from-scratch)")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
