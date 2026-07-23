import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env
from realism import apply_realism   # réalisme sim->réel OBLIGATOIRE (specs Unitree + DR)


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        # SAUVEGARDE FINE (tous les 25 iters, demande 22/07) : le pic-puis-effondrement
        # d'un run warm-starté peut être RAPIDE (escalier : pic ~iter 45, effondré <60
        # iters plus loin) → à 100 le vrai pic tombe ENTRE deux checkpoints et est perdu.
        # 25 garantit qu'on rend toujours près du sommet. Coût disque : ~4,5 Mo/ckpt.
        "save_interval": 25,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # Angles articulaires de repos (rad). Ordre : hanche / cuisse / mollet, x4 pattes.
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.4,
            "FR_calf_joint": -1.4,
            "RL_calf_joint": -1.4,
            "RR_calf_joint": -1.4,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,               # raideur PD
        "kd": 0.5,                # amortissement PD
        "termination_if_roll_greater_than": 10,   # deg
        "termination_if_pitch_greater_than": 10,  # deg
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "zero_cmd_frac": 0.15,          # 15% des envs en commande NULLE -> apprend l'arrêt figé
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.38,        # se tient HAUT et droit (défaut 0.30 = accroupi)
        "feet_height_target": 0.075,
        # Recette « MARCHE BIEN DROITE + ARRÊT FIGÉ » (inspirée du marcheur robust,
        # calibrée pour un départ FROM-SCRATCH sans warm-start) :
        #   orientation   -> tronc à plat (la clé du "droit")
        #   base_height   -> se lève (stance haute)
        #   hip_straight  -> pattes verticales sous le corps (pas "en biais")
        #   stand_still   -> base immobile à l'arrêt
        #   dof_freeze_stop -> pattes figées à l'arrêt (ne piétine plus)
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -0.5,
            "base_height": -150.0,
            "orientation": -3.0,
            "hip_straight": -1.0,
            "action_rate": -0.005,
            "similar_to_default": -0.05,
            "stand_still": -2.0,
            "dof_freeze_stop": -0.03,
        },
    }
    command_cfg = {
        "num_commands": 3,
        # Marche DIRIGEABLE : la policy apprend a obeir a n'importe quelle commande
        # (avant/arriere, pas lateraux, rotation) — necessaire pour les missions
        # pilotees (approche d'objet, orbite, retour). (0,0,0) = s'arreter.
        "lin_vel_x_range": [-0.5, 1.0],
        "lin_vel_y_range": [-0.4, 0.4],
        "ang_vel_range": [-0.8, 0.8],
    }

    # RÉALISME SIM->RÉEL appliqué PAR CONSTRUCTION (couple/vitesses bornés specs
    # Unitree + friction/poussées/charge randomisées). Vaut pour TOUT entraînement
    # du projet — source unique dans realism.py. Ne jamais retirer.
    apply_realism(env_cfg, reward_cfg)
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("--seed", type=int, default=1)
    # REPRISE : reprend depuis logs/<exp>/model_<N>.pt et PROLONGE de max_iterations
    # de plus (SANS effacer les logs) → la courbe TensorBoard continue proprement.
    # Sert quand la récompense monte encore à la fin d'un run (« laisse courir »).
    parser.add_argument("--resume_from", type=int, default=0,
                        help="itération de checkpoint à reprendre (0 = départ de zéro, efface les logs)")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    # SELF-COLLISION (correctif 22/07, revue vidéo) : sans elle, les pattes du marcheur
    # se TRAVERSENT — physiquement faux, et le vrai Go2 s'entrechoquerait. Comme le
    # marcheur est la base de TOUT (warm-starts, locomotion gelée du LiDAR), son défaut
    # se propageait à toute la chaîne. links_to_keep préserve les liens de pieds.
    env_cfg["self_collision"] = True
    env_cfg["links_to_keep"] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    train_cfg = get_train_cfg(args.exp_name)
    resume = args.resume_from > 0

    if not resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)          # départ de zéro : on repart propre
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
    # en reprise : on GARDE log_dir + cfgs.pkl + checkpoints existants (aucune perte)

    # Backend AMD ROCm explicite (fallback sur l'auto-selection gs.gpu si indisponible).
    backend = getattr(gs, "amdgpu", None) or gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if resume:
        ckpt = f"{log_dir}/model_{args.resume_from}.pt"
        runner.load(ckpt)                    # restaure poids + compteur d'itération
        print(f"REPRISE depuis {ckpt} (itération {args.resume_from}) → +{args.max_iterations} itérations")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()


"""
# entrainement (venv activé, GPU disponible) :
python go2_train.py -e go2-walking -B 4096 --max_iterations 2500

# reprendre / prolonger un entrainement existant :
python go2_train.py -e go2-walking --resume_from 2499 --max_iterations 1000

# suivi tensorboard :
tensorboard --logdir logs
"""
