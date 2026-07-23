"""RÉALISME SIM → RÉEL — source unique de vérité, OBLIGATOIRE pour TOUS les entraînements.

Règle du projet : chaque simulation doit être la plus réaliste possible (specs Unitree
Go2 + randomisation de domaine). Pour ne PAS dépendre de la mémoire humaine, ces
contraintes sont centralisées ici et appliquées par construction :

    from realism import apply_realism
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    apply_realism(env_cfg, reward_cfg)          # <- une ligne, impossible à oublier

Toute nouvelle compétence (courir, se coucher, obstacles…) DOIT appeler apply_realism.
"""

# Specs moteur réel Unitree Go2 (GO-M8010-6). Ordre = joint_names (hanche, cuisse,
# genou) × 4 pattes. Couples nominaux (crête ~45 N·m) ; vitesses max articulaires.
MOTOR_EFFORT_LIMITS = [23.7, 23.7, 35.55] * 4     # N·m
MOTOR_VEL_LIMITS = [30.1, 30.1, 20.07] * 4        # rad/s

# Clés env_cfg à activer sur CHAQUE entraînement (randomisation de domaine + moteur).
REALISM_ENV_CFG = {
    "enforce_motor_limits": True,          # couple borné aux specs moteur
    "motor_effort_limits": MOTOR_EFFORT_LIMITS,
    "dof_vel_limits": MOTOR_VEL_LIMITS,
    # le firmware du vrai Go2 CLIPPE toute consigne de position hors des limites
    # articulaires URDF (hanche ±60°, cuisse [-90°,200°]/[-30°,260°], genou
    # [-156°,-48°]) — on reproduit ce clip par construction (audit 21/07, URDF
    # officiel Unitree vérifié : efforts 23,7/23,7/35,55 N·m, vitesses
    # 30,1/30,1/20,07 rad/s, identiques aux constantes ci-dessus).
    "clamp_targets_to_limits": True,
    "friction_range": (0.4, 1.6),          # sol glissant -> adhérent, par épisode
    "push_interval_s": 4.0,                # bousculade toutes les ~4 s
    "push_vel": 1.0,                       # impulsion horizontale jusqu'à 1 m/s
    "payload_rand_kg": 0.5,                # charge dorsale randomisée 0-0,5 kg
    # bruit de capteurs (IMU + encodeurs) au niveau STANDARD legged_gym — référence
    # sim→réel éprouvée (gyro 0,2 rad/s ; gravité 0,05 ; encodeur pos 0,01 rad ;
    # encodeur vitesse 1,5 rad/s). RÈGLE PROJET (21/07) : on NE RÉDUIT JAMAIS le bruit
    # pour débloquer l'apprentissage — le bruit fait partie du réalisme et reste plein.
    # Si un entraînement FROM-SCRATCH plateaute, le bon levier est de RELÂCHER
    # l'exigence de POSTURE DROITE (base_height / hip_straight), pas de couper le bruit.
    # (Un plateau reward ~5 avait été observé avec bruit plein + posture TROP agressive
    # base_height -150 ; solution = adoucir la posture, garder le bruit.)
    "obs_noise": {"ang_vel": 0.2, "gravity": 0.05, "dof_pos": 0.01, "dof_vel": 1.5},
}

# Récompense qui garde les vitesses articulaires dans les specs constructeur.
REALISM_REWARD_SCALES = {"dof_vel_limit": -0.02}


def apply_realism(env_cfg, reward_cfg):
    """Active le réalisme sim→réel sur une config d'entraînement (in-place).
    Idempotent, sûr à appeler sur n'importe quel env_cfg/reward_cfg du projet."""
    env_cfg.update(REALISM_ENV_CFG)
    reward_cfg.setdefault("reward_scales", {}).update(REALISM_REWARD_SCALES)
    return env_cfg, reward_cfg
