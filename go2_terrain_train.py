"""Entraînement LOCOMOTION TOUT-TERRAIN — robot NU (segment « franchir obstacles »).

Warm-start depuis le marcheur `go2-walking` (même obs 45-D proprioceptive → transfert
direct). Grille de terrains mixtes, du plus facile au plus dur, COMPLEXIFIÉE par rapport
à la version archivée : grille 5×5 (au lieu de 4×4), pentes plus raides, obstacles /
marches plus hauts, escaliers pyramidaux. Le robot déjà bon marcheur affronte le relief.

Priorités (cohérentes avec la règle projet « stabilité > posture droite » sur tout
régime non-plat) :
  - STABILITÉ d'abord : `orientation` (tronc à plat) fort — c'est la clé pour ne pas
    basculer sur une pente / une marche. La posture « haute et droite » est secondaire.
  - HAUTEUR RELATIVE au terrain (override dans l'env) et DOUCE : sur le relief la garde
    au sol varie, elle ne doit pas écraser le suivi de vitesse.
  - FRANCHISSEMENT : `feet_air_time` récompense de vrais pas (lever les pattes) pour
    passer marches et obstacles.
  - ARRÊT MAÎTRISÉ SUR PENTE : `zero_cmd_frac` + `stand_still` + `dof_freeze_stop` →
    le robot doit s'immobiliser stable même sur un plan incliné (demandé 21/07).
  - RÉALISME sim→réel par construction via `realism.apply_realism` (couples/vitesses
    Unitree bornés + friction/poussées/charge + BRUIT CAPTEURS PLEIN — jamais réduit).

    python go2_terrain_train.py -e go2-terrain --max_iterations 2500
    python go2_terrain_train.py -e go2-terrain --resume_from <N> --max_iterations 1000
"""
import argparse
import glob
import os
import pickle
import re
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_terrain_env import Go2TerrainEnv
from go2_train import get_train_cfg, get_cfgs
from realism import apply_realism

WARM_EXP = "go2-walking"                 # marcheur droit réaliste = base du tout-terrain
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Grille 5×5 de sous-terrains — LIGNE = difficulté croissante, COLONNE = variante.
# Types tous « pleins » (pas de trous à vide type stepping_stones : principale source
# d'éjections/glitchs qui empoisonnaient l'apprentissage — à réintroduire seulement
# après validation du build). Complexification = grille plus large + relief plus dur.
SUBTERRAINS = [
    ["flat_terrain", "random_uniform_terrain", "sloped_terrain", "wave_terrain", "random_uniform_terrain"],
    ["random_uniform_terrain", "wave_terrain", "sloped_terrain", "discrete_obstacles_terrain", "sloped_terrain"],
    ["discrete_obstacles_terrain", "pyramid_sloped_terrain", "wave_terrain", "sloped_terrain", "random_uniform_terrain"],
    ["stairs_terrain", "discrete_obstacles_terrain", "pyramid_sloped_terrain", "pyramid_stairs_terrain", "sloped_terrain"],
    ["pyramid_stairs_terrain", "stairs_terrain", "discrete_obstacles_terrain", "pyramid_sloped_terrain", "stairs_terrain"],
]


# --- terrains MONO-TYPE (un segment vidéo par obstacle, demande 21/07) ---
# Chaque famille = un seul type d'obstacle, grille 4×4 randomisée (variété de difficulté).
# Bien plus lisible que le mixte : le robot apprend UN obstacle proprement (rate au début
# sur CE type → le maîtrise), pas un échec brouillon sur des tuiles dures tirées au hasard.
FAMILIES = {
    "pente":     ("sloped_terrain",             {"slope": 0.20}),                         # ~11° (17° faisait culbuter le marcheur warm-flat → divergence, audit 22/07)
    "escalier":  ("stairs_terrain",             {"step_width": 0.35, "step_height": 0.05}),  # 7cm→5cm (revue vidéo 22/07 : le marcheur warm-flat tombait) + marche plus large
    "pyramide":  ("pyramid_stairs_terrain",     {"step_width": 0.35, "step_height": 0.05}),  # idem
    # NB escalier/pyramide : heightfield à MAILLE FINE obligatoire (voir FAMILY_HSCALE) —
    # à 0,25 m de maille, la contremarche de 7 cm est interpolée sur 25 cm → une RAMPE
    # lisse, pas un escalier (constaté sur les rendus du 22/07). À 0,05 m, la marche est
    # quasi verticale (7 cm sur 5 cm de course) : vraie géométrie, vraie difficulté.
    "accidente": ("random_uniform_terrain",     {"min_height": -0.10, "max_height": 0.10,
                                                 "step": 0.02, "downsampled_scale": 0.5}),
    "rebords":   ("discrete_obstacles_terrain", {"max_height": 0.10, "min_size": 0.5,
                                                 "max_size": 2.0, "num_rects": 20}),   # 16→10cm (marche verticale) par précaution warm-flat
    # TROUS / passages à ne pas rater : dalles séparées par des vides de 10 cm,
    # profonds de 25 cm (nid-de-poule / caillebotis réaliste — pas un gouffre de
    # 10 m qui gâcherait la moitié des spawns et l'apprentissage).
    "trous":     ("stepping_stones_terrain",    {"stone_size": 0.45, "stone_distance": 0.10,
                                                 "max_height": 0.02, "platform_size": 1.5,
                                                 "depth": -0.25}),
    # VAGUES : houle douce de 12 cm d'amplitude — le sol « roule » sous le robot
    # (complément lisse de l'accidenté, qui lui est anguleux).
    "vagues":    ("wave_terrain",               {"num_waves": 3, "amplitude": 0.12}),
}


# maille horizontale du heightfield PAR FAMILLE : fine pour les marches (contremarches
# verticales), standard (0,25 m) pour le relief continu (pente/vagues/accidenté).
FAMILY_HSCALE = {"escalier": 0.05, "pyramide": 0.05}


def single_terrain_cfg(family):
    typ, params = FAMILIES[family]
    return dict(
        n_subterrains=(4, 4),
        subterrain_size=(9.0, 9.0),
        horizontal_scale=FAMILY_HSCALE.get(family, 0.25),
        vertical_scale=0.005,
        subterrain_types=[[typ] * 4 for _ in range(4)],
        randomize=True,
        subterrain_parameters={typ: params},
    )


def terrain_cfg():
    return dict(
        n_subterrains=(5, 5),
        subterrain_size=(9.0, 9.0),
        horizontal_scale=0.25,
        vertical_scale=0.005,
        subterrain_types=SUBTERRAINS,
        randomize=True,
        # `slope` = tangente de l'angle. Défaut Genesis ~27° = trop raide pour un
        # quadrupède. Ici plus raide que l'archive (0.30) mais encore marchable :
        subterrain_parameters={
            "sloped_terrain": {"slope": 0.35},          # ~19° (montée en +x)
            "pyramid_sloped_terrain": {"slope": 0.25},  # ~14° au plus raide
            # rebords/obstacles PLUS HAUTS que l'archive (0.13 → 0.17 m) et plus nombreux
            "discrete_obstacles_terrain": {
                "max_height": 0.17, "min_size": 0.5, "max_size": 2.0, "num_rects": 20,
            },
            # marches franches (hauteur ~marche d'escalier réelle)
            "stairs_terrain": {"step_width": 0.30, "step_height": 0.13},
            "pyramid_stairs_terrain": {"step_width": 0.30, "step_height": 0.11},
        },
    )


def last_ckpt(exp):
    its = [int(m.group(1)) for f in glob.glob(f"logs/{exp}/model_*.pt")
           if (m := re.search(r"model_(\d+)\.pt$", f))]
    return max(its) if its else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", default="")
    p.add_argument("-B", "--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=1200)
    p.add_argument("--family", default="", choices=[""] + list(FAMILIES),
                   help="type d'obstacle UNIQUE (pente/escalier/pyramide/accidente/rebords). "
                        "Vide = terrain mixte 5×5. Un segment vidéo par famille.")
    p.add_argument("--init_ckpt", default="", help="warm start explicite (défaut : marcheur)")
    p.add_argument("--resume_from", type=int, default=0, help="reprendre/prolonger un run")
    p.add_argument("--save_interval", type=int, default=0,
                   help="sauvegarde tous les N iters (0 = défaut du train_cfg). L'escalier "
                        "PIQUE vers l'itération ~45 puis s'effondre en <60 iters : à 100 le "
                        "pic tombe ENTRE deux checkpoints et est PERDU (constat 22/07) → 25.")
    args = p.parse_args()
    # nom d'expérience auto : go2-terrain-<famille> (ou go2-terrain pour le mixte)
    if not args.exp_name:
        args.exp_name = f"go2-terrain-{args.family}" if args.family else "go2-terrain"

    gs.init(backend=getattr(gs, "amdgpu", None) or gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["terrain_cfg"] = single_terrain_cfg(args.family) if args.family else terrain_cfg()
    env_cfg["self_collision"] = True
    env_cfg["links_to_keep"] = FEET
    env_cfg["termination_if_pitch_greater_than"] = 55   # relief : tangage légitime
    env_cfg["termination_if_roll_greater_than"] = 55
    # ARRÊT MAÎTRISÉ SUR PENTE : une fraction des essais reçoit la commande nulle → le
    # robot apprend à s'immobiliser stable sur le relief, pas seulement à le franchir.
    env_cfg["zero_cmd_frac"] = 0.15    # + d'essais à l'arrêt : maîtriser l'immobilisation sur CHAQUE type

    # vitesses commandées MODÉRÉES : le tout-terrain n'est pas un sprint ; on veut du
    # franchissement contrôlé (pas de curriculum de vitesse ici, contrairement à la course).
    command_cfg["lin_vel_x_range"] = [-1.0, 1.5]
    command_cfg["lin_vel_y_range"] = [-0.5, 0.5]
    command_cfg["ang_vel_range"] = [-1.0, 1.0]

    reward_cfg["tracking_sigma"] = 0.25
    reward_cfg["base_height_target"] = 0.32             # RELATIF au terrain (override env)
    # STABILITÉ > POSTURE DROITE (règle projet) : orientation forte, hauteur douce,
    # pas de terme hip_straight (liberté de posture pour franchir).
    reward_cfg["reward_scales"] = {
        "tracking_lin_vel": 1.0,        # suivre la vitesse commandée
        "tracking_ang_vel": 0.2,
        "lin_vel_z": -0.3,              # doux : grimper demande du mouvement vertical
        "base_height": -6.0,            # RELATIF terrain, TRÈS doux (posture = secondaire)
        "orientation": -3.5,            # STABILITÉ renforcée : tronc à plat = PRIORITÉ
        "feet_air_time": 0.6,           # vrais pas → franchit marches/obstacles
        "undesired_contact": -1.2,      # stabilité : pas de contacts parasites (trébuche)
        "action_rate": -0.008,          # gestes plus doux = plus stable
        # POSTURE DROITE RELÂCHÉE AU MINIMUM (demande 21/07) : pas de hip_straight,
        # similar_to_default quasi nul → liberté TOTALE de posture pour encaisser le relief.
        "similar_to_default": -0.005,
        # ARRÊT MAÎTRISÉ sur CHAQUE type : base immobile + PATTES FIGÉES (renforcé) à
        # commande nulle → le robot tient sa position sans piétiner, même sur pente.
        "stand_still": -2.5,
        "dof_freeze_stop": -0.09,       # FORT : à l'arrêt, les pattes ne bougent plus
    }
    # ESCALIER / PYRAMIDE — override (revue vidéo 22/07 : le robot TOMBAIT sur les
    # marches). Cause : la pénalité `orientation` (tronc à plat, -3.5) COMBAT le tangage
    # nécessaire pour grimper une contremarche (le robot doit cabrer l'avant pour poser
    # une patte sur la marche du dessus). On la RELÂCHE fortement et on RENFORCE
    # `feet_air_time` (lever franchement les pattes = poser sur la marche) + on relâche
    # un peu `undesired_contact` (les pattes touchent les arêtes des marches, normal).
    if args.family in ("escalier", "pyramide"):
        reward_cfg["reward_scales"]["orientation"] = -0.8      # -3.5 → tangage LIBRE pour grimper
        reward_cfg["reward_scales"]["feet_air_time"] = 1.5     # lever franc des pattes sur la marche
        reward_cfg["reward_scales"]["undesired_contact"] = -0.5  # arêtes de marche tolérées
        reward_cfg["reward_scales"]["lin_vel_z"] = -0.1        # grimper = beaucoup de mouvement vertical

    # RÉALISME sim→réel PAR CONSTRUCTION (dof_vel_limit + couples bornés + friction/
    # poussées/charge/bruit PLEIN). On ne réduit JAMAIS le bruit (règle projet).
    apply_realism(env_cfg, reward_cfg)

    train_cfg = get_train_cfg(args.exp_name)
    if args.save_interval > 0:
        train_cfg["save_interval"] = args.save_interval
    log_dir = f"logs/{args.exp_name}"
    resume = args.resume_from > 0
    if not resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    env = Go2TerrainEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                        reward_cfg=reward_cfg, command_cfg=command_cfg)
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
