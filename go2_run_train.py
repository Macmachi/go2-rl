"""Entraînement COURSE — trot/galop rapide et stable sur sol plat (robot nu).

Warm-start depuis le marcheur `go2-walking` : le robot sait déjà marcher droit, on
l'amène au SPRINT via un curriculum de vitesse. Sol plat volontaire — la course tout-
terrain est un segment distinct (« franchir obstacles »). Points clés (leçons projet
+ littérature locomotion RL) :

  - PRIORITÉ = STABILITÉ, PAS POSTURE DROITE. À la course, ce qui compte c'est que le
    robot ne bascule/tombe pas (tronc à plat via `orientation`), pas qu'il se tienne
    haut et droit. Les termes de posture droite (`base_height_slow`/`hip_straight_slow`)
    sont donc TRÈS relâchés ET bridés à < 1 m/s : une pénalité de hauteur forte ou
    inconditionnelle tue le sprint — le galop DOIT s'abaisser, et PPO « refuse alors de
    courir ». On garde juste assez de posture pour une station propre à l'arrêt.
  - BRUIT capteurs PLEIN conservé (réalisme) : on ne réduit jamais le bruit pour
    faciliter l'apprentissage — le warm-start marcheur suffit à l'absorber.
  - `feet_air_time` élevé : récompense une vraie foulée avec phase de vol plutôt qu'un
    piétinement traînant.
  - Curriculum vx 1.0 → vmax : sauter direct au max fait décrocher la policy.
  - v3 (21/07) : OBJECTIF LIMITE CONSTRUCTEUR. Le Go2 EDU est donné pour ~5 m/s
    (test labo Unitree). On repart du sprinteur 2,7 m/s (warm go2-run) et on rampe
    vers 5,0 en libérant la course À RAS DU SOL : orientation allégée (tangage de
    galop légitime), lin_vel_z allégé (rebond de foulée), posture droite déjà
    bridée à < 1 m/s. La vitesse ATTEINTE (bornée par les couples/vitesses réels
    des moteurs) fait foi — c'est elle qu'on mesure et qu'on montre.
  - Réalisme sim→réel HÉRITÉ (couple/vitesses bornés Unitree + friction/poussées/
    charge/bruit capteurs) via `realism.apply_realism` — la vitesse atteinte est donc
    plafonnée par la PHYSIQUE réelle des moteurs, pas par un idéal.

    python go2_run_train.py -e go2-run --max_iterations 3000
    python go2_run_train.py -e go2-run --resume_from 2999 --max_iterations 1000
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

WARM_EXP = "go2-walking"                 # marcheur droit réaliste = base du sprint
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def last_ckpt(exp):
    its = [int(m.group(1)) for f in glob.glob(f"logs/{exp}/model_*.pt")
           if (m := re.search(r"model_(\d+)\.pt$", f))]
    return max(its) if its else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", default="go2-run")
    p.add_argument("-B", "--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=3000)
    p.add_argument("--vmax", type=float, default=5.0,
                   help="vitesse avant max COMMANDÉE (m/s). 5,0 = limite constructeur du "
                        "Go2 EDU (« max running speed ~5 m/s », testé en labo par Unitree). "
                        "Le robot atteint ce que la physique bornée (couples nominaux "
                        "23,7/35,55 N·m, vitesses 30,1/20,07 rad/s) permet — la vitesse "
                        "ATTEINTE fait foi, pas la commande.")
    p.add_argument("--vstart", type=float, default=0.0,
                   help="départ du curriculum de vitesse (m/s). 0 = auto : 1,0 en warm "
                        "marcheur, vmax en reprise --init_ckpt sprint. Ex. 2,7 pour repartir "
                        "du palier stable de go2-run.")
    p.add_argument("--ramp", type=int, default=25000,
                   help="pas de rampe du curriculum de vitesse (plus long = plus doux, "
                        "consolide la stabilité à chaque palier avant d'accélérer).")
    p.add_argument("--init_ckpt", default="", help="warm start explicite")
    p.add_argument("--resume_from", type=int, default=0, help="reprendre/prolonger un run")
    p.add_argument("--zero_frac", type=float, default=0.12,
                   help="fraction d'essais à commande nulle (0.3 = finetune d'arrêt renforcé)")
    args = p.parse_args()

    gs.init(backend=getattr(gs, "amdgpu", None) or gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["self_collision"] = True
    env_cfg["links_to_keep"] = FEET                     # pieds séparés -> feet_air_time
    env_cfg["termination_if_pitch_greater_than"] = 55   # course : tangage légitime
    env_cfg["termination_if_roll_greater_than"] = 55
    # ARRÊT MAÎTRISÉ APRÈS SPRINT : une fraction des essais reçoit la commande nulle.
    # Comme les commandes se retirent toutes les ~4 s, un env qui SPRINTAIT reçoit
    # soudain « stop » → il apprend la TRANSITION course→décélération→immobilisation
    # stable. 12 % s'est avéré INSUFFISANT (rendu 22/07 : coupure 2,7→0 = SALTO avant —
    # l'inertie le fait basculer, la transition est trop rare à l'entraînement) →
    # --zero_frac permet un finetune d'arrêt renforcé (ex. 0.3, reprise courte).
    env_cfg["zero_cmd_frac"] = args.zero_frac
    # curriculum vitesse : de la vitesse du marcheur (1.0) au sprint (vmax). En reprise
    # warm (init_ckpt), --vstart permet de repartir du palier déjà maîtrisé (ex. 2,7).
    env_cfg["vx_cap_start"] = args.vstart or (args.vmax if args.init_ckpt else 1.0)
    env_cfg["vx_cap_end"] = args.vmax
    env_cfg["vx_ramp_steps"] = args.ramp

    command_cfg["lin_vel_x_range"] = [-1.0, args.vmax]
    command_cfg["lin_vel_y_range"] = [-0.5, 0.5]
    command_cfg["ang_vel_range"] = [-1.0, 1.0]

    reward_cfg["tracking_sigma"] = 0.35                 # plus large : près de la limite physique, une commande
                                                        # partiellement atteinte doit RESTER récompensée (sinon le
                                                        # gradient s'effondre quand on commande la limite constructeur)
    reward_cfg["base_height_target"] = 0.37
    # PRIORITÉ COURSE = STABILITÉ, PAS POSTURE DROITE (consigne 21/07). À la course, exiger
    # une station haute / des pattes bien droites est contre-productif : le galop DOIT
    # s'abaisser et engager les hanches. On distingue donc deux familles de récompenses :
    #   - POSTURE DROITE (base_height, hip_straight) : très RELÂCHÉE ici (et déjà bridée
    #     à < 1 m/s via les variantes *_slow, donc quasi inactive en sprint) — juste assez
    #     pour une station propre à l'arrêt, jamais assez pour brider la foulée.
    #   - STABILITÉ (orientation = tronc à plat/ne bascule pas ; lin_vel_z ; ne pas tomber
    #     via undesired_contact) : GARDÉE forte. C'est elle qui rend la course crédible.
    # Le BRUIT capteurs reste PLEIN (hérité de realism.apply_realism) — on ne le coupe pas.
    reward_cfg["reward_scales"] = {
        "tracking_lin_vel": 1.6,        # suivre la vitesse = priorité
        "tracking_ang_vel": 0.3,
        "lin_vel_z": -0.15,             # ALLÉGÉ (v3) : le galop à ras du sol REBONDIT — trop pénaliser
                                        # l'oscillation verticale bride la foulée près de la limite
        "base_height_slow": -40.0,      # posture haute TRÈS relâchée, et seulement < 1 m/s (pour l'arrêt)
        "hip_straight_slow": -0.3,      # pattes droites : exigence minimale, < 1 m/s seulement
        "orientation": -1.5,            # ALLÉGÉ (v3, consigne « courir à ras le sol ») : le sprint engage
                                        # un tangage dynamique légitime ; la non-chute reste garantie par
                                        # la terminaison 55°, undesired_contact et lin_vel_z
        "feet_air_time": 1.0,           # vraie foulée (phase de vol)
        "undesired_contact": -1.0,      # stabilité : pas de contacts parasites (genoux/tronc au sol)
        "action_rate": -0.005,
        "similar_to_default": -0.005,   # très bas : la course exige de la liberté de posture
        # ARRÊT MAÎTRISÉ (après sprint) : base immobile + pattes figées à commande nulle.
        "stand_still": -2.0,            # base immobile (translation + lacet)
        "dof_freeze_stop": -0.03,       # pattes qui cessent de bouger (ne piétine plus)
    }
    # RÉALISME sim->réel PAR CONSTRUCTION (ré-injecte dof_vel_limit + garde les clés env).
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

    env = Go2Env(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if resume:
        runner.load(f"{log_dir}/model_{args.resume_from}.pt")
        print(f"REPRISE model_{args.resume_from} → +{args.max_iterations} itérations")
    else:
        warm = args.init_ckpt or (f"logs/{WARM_EXP}/model_{last_ckpt(WARM_EXP)}.pt"
                                  if last_ckpt(WARM_EXP) else "")
        if warm and os.path.exists(warm):
            runner.load(warm)
            print(f"warm start depuis {warm}")
        else:
            print("(pas de warm start — départ de zéro)")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
