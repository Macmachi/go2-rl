"""SEGMENT LiDAR — ÉVITEMENT APPRIS d'un parcours d'obstacles FIXES + MOBILES (robot nu).

Système HIÉRARCHIQUE : une policy de NAVIGATION (obs 80-D, 3 actions vx/vy/yaw @ 10 Hz)
pilote la policy de LOCOMOTION GELÉE (le marcheur, @ 50 Hz). La nav ne « voit » le monde
qu'à travers un LiDAR 72 secteurs (type Unitree L1) — jamais la vérité terrain, qui ne
sert qu'aux récompenses (privilège d'entraînement). Tâche : rejoindre un BUT droit devant
en CONTOURNANT des obstacles, dont certains SE DÉPLACENT (piétons qui traversent).

Deux couches de sûreté combinées (demande 21/07) :
  - Couche A (sécurité) : collision = terminaison + forte pénalité (le robot DOIT éviter).
  - Couche B (évitement appris) : pénalité de PROXIMITÉ croissante quand la garde
    robot↔obstacle se réduit → la policy apprend à s'écarter AVANT le contact.

Scan analytique (ray-cercle, torch GPU) contre la pose RÉELLE de chaque obstacle → suit
automatiquement les obstacles mobiles. Obstacles = cylindres (poteaux / piétons), scan
robuste (pas de fenêtre verticale fragile). Rendu des rayons : voir le script de rendu.
"""
import math

import torch
from tensordict import TensorDict

import genesis as gs

from go2_env import Go2Env

# ---- LiDAR (géométrie type Unitree L1, reprise du proto validé) ----
N_H = 72                       # 72 secteurs horizontaux
LIDAR_MAX = 6.5                # portée (m)
CHASSIS_MASK = 0.45            # échos < 45 cm = pattes du robot → ignorés

# ---- navigation ----
NAV_DECIM = 5                  # 1 action nav = 5 pas de locomotion (10 Hz)
EPISODE_S = 40.0               # durée max de mission
GOAL_X = 9.0                   # but droit devant à ~9 m
GOAL_Y_RANGE = 1.2             # décalage latéral aléatoire du but (dans le couloir muré)
ARRIVE = 0.7                   # distance de succès au but
CMD_LOW = torch.tensor([-0.5, -0.4, -0.8])    # vx, vy, yaw commandés à la locomotion
CMD_HIGH = torch.tensor([1.2, 0.4, 0.8])      # (biaisé vers l'avant)

# ---- parcours d'obstacles : cylindres (rayon m, hauteur m, mobile ?) ----
# v2 (21/07) : 8 obstacles répartis dans le couloir (3 MOBILES imprévisibles) + une
# BARRIÈRE de 9 poteaux avec un PASSAGE ÉTROIT à position aléatoire (les poteaux
# barrant la porte sont « garés » hors scène). Une rangée de cylindres rapprochés
# joue le rôle d'un mur : même scan rayon-cercle, même garde, et visuellement une
# vraie barrière au rendu — sans avoir à coder du rayon-segment.
# v3 (22/07) : COULOIR MURÉ. Les rendus v2 montraient le robot qui CONTOURNE tout le
# parcours par l'extérieur (trajectoire large hors champ = « triche », pas de l'évitement).
# → deux MURS de poteaux serrés ferment le couloir sur toute sa longueur (le robot DOIT
# se faufiler ENTRE les obstacles) + sortie de couloir = terminaison pénalisée. En
# compensation, les passages sont ÉLARGIS (obstacles recentrés |y| ≤ 1,2, rayons ≤ 0,28,
# porte du goulot ~1,3-1,9 m) : l'évitement reste faisable, le contournement ne l'est plus.
WALL_Y = 2.5                   # demi-largeur du couloir (murs à y = ±2,5)
N_WALL = 20                    # poteaux de mur par côté (interstices 23 cm : infranchissables)
WALL_X0, WALL_DX = -0.6, 0.55  # le mur couvre x ∈ [-0,6 ; 9,85]
OBSTACLES = [
    dict(r=0.25, h=1.0, moving=False),   # poteau
    dict(r=0.28, h=1.0, moving=False),
    dict(r=0.22, h=1.0, moving=True),    # piéton
    dict(r=0.28, h=1.0, moving=False),
    dict(r=0.22, h=1.0, moving=True),    # piéton
    dict(r=0.25, h=1.0, moving=False),
    dict(r=0.22, h=1.0, moving=True),    # piéton
    dict(r=0.26, h=1.0, moving=False),
] + [dict(r=0.16, h=0.9, moving=False, fence=True) for _ in range(9)] \
  + [dict(r=0.16, h=0.9, moving=False, wall=True) for _ in range(2 * N_WALL)]
N_OBST = len(OBSTACLES)
FENCE_Y = [-2.2 + 0.55 * j for j in range(9)]   # grille des poteaux (interstices 23 cm : infranchissables)
FENCE_GAP = 0.80               # demi-largeur de la porte ÉLARGIE : ~1,3-1,9 m libre (v2 : 0,8-1,3)
MOVE_AMP = 1.2                  # amplitude latérale des piétons (m) — borne haute, tirée par épisode
# fréquence d'oscillation (Hz) — vitesse de POINTE d'un piéton = 2π·f·A. Bornée pour
# rester à une allure de MARCHE humaine (~0,6-1,4 m/s de pointe) : l'ancien tirage
# (jusqu'à 0,35 Hz × 1,2 m ≈ 2,6 m/s) faisait des piétons au sprint (revue 22/07).
MOVE_FREQ = 0.13               # valeur centrale, tirée par épisode dans [0.08, 0.18]
COLLIDE_MARGIN = 0.28          # rayon effectif du robot (chute si garde surface < 0)
LIDAR_NOISE_STD = 0.03         # bruit gaussien de portée (m) — ordre L1 réel
LIDAR_DROPOUT = 0.03           # probabilité d'écho perdu par secteur (faux LIDAR_MAX)


def load_frozen_actor(ckpt_path, device):
    """Recharge le MLP acteur de la locomotion (45 → 12), gelé (couche B pilote celle-ci)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k[len("mlp."):]: v for k, v in ck["actor_state_dict"].items() if k.startswith("mlp.")}
    net = torch.nn.Sequential(
        torch.nn.Linear(45, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 12),
    ).to(device)
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class Go2LidarAvoidEnv:
    """Env RL de navigation-évitement (obs 80-D, actions 3-D) pour rsl-rl."""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 locomotion_ckpt, show_viewer=False, camera_cfg=None):
        self.num_envs = num_envs
        self.num_actions = 3
        self.cfg = env_cfg
        self.device = gs.device
        self.dt = 0.02
        self.nav_dt = self.dt * NAV_DECIM
        self.max_episode_length = math.ceil(EPISODE_S / self.nav_dt)
        self.reward_cfg = reward_cfg

        # obstacles physiques (cylindres kinématiques repositionnés au reset / à chaque pas)
        obst_morphs = []
        for i, o in enumerate(OBSTACLES):
            spawn = (14.0 + 2.0 * i, 14.0, o["h"] / 2)     # garés loin, placés au reset
            obst_morphs.append(gs.morphs.Cylinder(radius=o["r"], height=o["h"],
                                                  pos=spawn, fixed=True))
        # locomotion : env physique + policy gelée
        env_cfg = dict(env_cfg)
        env_cfg["episode_length_s"] = 10 * EPISODE_S       # la nav gère les épisodes
        env_cfg["resampling_time_s"] = 10 * EPISODE_S
        self.lo_env = Go2Env(num_envs, env_cfg, obs_cfg, {"reward_scales": {}},
                             command_cfg, show_viewer=show_viewer,
                             extra_morphs=obst_morphs, camera_cfg=camera_cfg)
        self.actor_lo = load_frozen_actor(locomotion_ckpt, self.device)
        self.obst = self.lo_env.extra_entities[:N_OBST]

        self.cmd_low = CMD_LOW.to(self.device)
        self.cmd_high = CMD_HIGH.to(self.device)

        # géométrie LiDAR (azimuts capteur, convention [-π, +π[)
        self._az = (-math.pi + torch.arange(N_H, dtype=gs.tc_float, device=self.device)
                    / N_H * 2 * math.pi).unsqueeze(0)                # (1, 72)
        # COUCHE A — arrêt d'urgence déterministe (voir _safety_filter). Secteurs
        # frontaux ±45° ; désactivée à l'entraînement (la couche B apprend sans
        # béquille), activée à l'inférence via env_cfg["nav_safety_stop"]=True.
        self.safety_stop = bool(env_cfg.get("nav_safety_stop", False))
        self._front_mask = torch.cos(self._az.squeeze(0)) > math.cos(math.radians(45.0))
        self._obst_r = torch.tensor([o["r"] for o in OBSTACLES],
                                    dtype=gs.tc_float, device=self.device)      # (N_OBST,)
        self._moving = torch.tensor([o["moving"] for o in OBSTACLES],
                                    dtype=torch.bool, device=self.device)       # (N_OBST,)
        self._obst_h = torch.tensor([o["h"] for o in OBSTACLES],
                                    dtype=gs.tc_float, device=self.device)

        Z = lambda *sh: torch.zeros(sh, dtype=gs.tc_float, device=self.device)
        self.goal = Z(num_envs, 2)
        self.obst_home = Z(N_OBST, num_envs, 2)     # position de base (les mobiles oscillent autour)
        self.obst_xy = Z(N_OBST, num_envs, 2)       # position courante (vérité pour garde/collision)
        self.obst_phase = Z(N_OBST, num_envs)       # déphasage individuel des piétons
        # piétons IMPRÉVISIBLES (v2) : fréquence, amplitude et DIRECTION d'oscillation
        # tirées par épisode — la policy ne peut pas apprendre par cœur une trajectoire.
        self.obst_freq = torch.full((N_OBST, num_envs), MOVE_FREQ, dtype=gs.tc_float, device=self.device)
        self.obst_amp = torch.full((N_OBST, num_envs), MOVE_AMP, dtype=gs.tc_float, device=self.device)
        self.obst_dir = Z(N_OBST, num_envs, 2)
        self.obst_dir[:, :, 1] = 1.0                # défaut : traversée latérale (axe y)
        self._fence_ids = [k for k, o in enumerate(OBSTACLES) if o.get("fence")]
        self._wall_ids = [k for k, o in enumerate(OBSTACLES) if o.get("wall")]
        # BRUIT LiDAR réaliste (portée + échos perdus) — actif par défaut (règle réalisme) ;
        # désactivable par env_cfg["lidar_noise"]=False (debug uniquement).
        self.lidar_noise = bool(env_cfg.get("lidar_noise", True))
        # SCÉNARIO de scène (pour les mini-segments vidéo — même policy, décors séparés) :
        #   "complet" (défaut, = entraînement) | "fixes"   (obstacles immobiles, pas de barrière)
        #   "pietons" (mobiles, pas de barrière) | "goulot" (barrière + fixes, pas de mobiles)
        self.scenario = str(env_cfg.get("lidar_scenario", "complet"))
        self.last_nav_action = Z(num_envs, 3)
        self.prev_dist = Z(num_envs)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.rew_buf = Z(num_envs)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.extras = {"observations": {}}
        self.episode_sums = {k: Z(num_envs) for k in
                             ["progress", "clearance", "collision", "success", "fall",
                              "heading", "action", "escape"]}
        self._ready = True
        self.reset()

    # ------------------------------------------------------------------ utils
    def _base_xy_yaw(self):
        pos = self.lo_env.base_pos[:, :2]
        q = self.lo_env.base_quat
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        return pos, yaw

    def _move_obstacles(self, moving_only=True):
        """Position courante des obstacles : les mobiles oscillent le long d'une DIRECTION
        propre à l'épisode (fréquence/amplitude/cap tirés au reset — piétons imprévisibles
        d'un épisode à l'autre, lisses dedans). Déterministe (compteur de pas). Par défaut
        ne repositionne physiquement que les MOBILES (les fixes ne bougent pas → set_pos
        seulement au reset, via moving_only=False) — divise le coût de synchro GPU par pas."""
        t = self.episode_length_buf.to(gs.tc_float) * self.nav_dt          # (N,)
        osc = self.obst_amp * torch.sin(2 * math.pi * self.obst_freq * t.unsqueeze(0)
                                        + self.obst_phase)                 # (N_OBST,N)
        mv = self._moving.unsqueeze(1).to(gs.tc_float)
        self.obst_xy = self.obst_home.clone()
        self.obst_xy[:, :, 0] += mv * osc * self.obst_dir[:, :, 0]
        self.obst_xy[:, :, 1] += mv * osc * self.obst_dir[:, :, 1]
        for k, ent in enumerate(self.obst):
            if moving_only and not bool(self._moving[k]):
                continue
            z = torch.full((self.num_envs, 1), float(self._obst_h[k]) / 2,
                           dtype=gs.tc_float, device=self.device)
            ent.set_pos(torch.cat([self.obst_xy[k], z], dim=1))

    def _clearance(self, pos):
        """Garde surface-à-surface minimale robot↔obstacle (m), par env (vérité terrain)."""
        d = torch.linalg.norm(self.obst_xy - pos.unsqueeze(0), dim=2)       # (N_OBST, N)
        return (d - self._obst_r.unsqueeze(1)).min(dim=0).values             # (N,)

    def read_scan(self):
        """Scan LiDAR analytique (N, 72) : ray-cercle contre chaque cylindre, min par
        secteur. Origine = tête du robot ; masque châssis ; borné à LIDAR_MAX."""
        pos, yaw = self._base_xy_yaw()
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        ox = (pos[:, 0] + 0.28 * cy).unsqueeze(1)          # (N,1) origine capteur (tête)
        oy = (pos[:, 1] + 0.28 * sy).unsqueeze(1)
        az = self._az + yaw.unsqueeze(1)                    # (N,72) azimuts monde
        dx, dy = torch.cos(az), torch.sin(az)               # (N,72) directions unitaires
        scan = torch.full((self.num_envs, N_H), LIDAR_MAX, dtype=gs.tc_float, device=self.device)
        for k in range(N_OBST):
            cxk = self.obst_xy[k, :, 0].unsqueeze(1)        # (N,1)
            cyk = self.obst_xy[k, :, 1].unsqueeze(1)
            ocx, ocy = ox - cxk, oy - cyk                   # (N,1) origine - centre
            b = dx * ocx + dy * ocy                         # (N,72) proj (d unitaire)
            c = (ocx * ocx + ocy * ocy) - self._obst_r[k] ** 2   # (N,1)
            disc = b * b - c                                # (N,72)
            hit = disc > 0
            t = -b - torch.sqrt(disc.clamp(min=0.0))        # racine la plus proche
            hit = hit & (t > 0)
            dk = torch.where(hit, t, torch.full_like(t, LIDAR_MAX))
            scan = torch.minimum(scan, dk)
        scan = torch.where(scan < CHASSIS_MASK, torch.full_like(scan, LIDAR_MAX), scan)
        # BRUIT capteur réaliste (v2, règle réalisme max) : bruit gaussien de portée
        # (~3 cm, ordre L1) + échos PERDUS aléatoires (secteur → LIDAR_MAX, comme une
        # absorption/réflexion spéculaire). Appliqué à TOUS les consommateurs du scan
        # (policy nav, couche A, rendu) — le vrai capteur ne fait pas de faveurs.
        if self.lidar_noise:
            scan = scan + torch.randn_like(scan) * LIDAR_NOISE_STD
            drop = torch.rand_like(scan) < LIDAR_DROPOUT
            scan = torch.where(drop, torch.full_like(scan, LIDAR_MAX), scan)
        out = scan.clip(min=0.0, max=LIDAR_MAX)
        # cache du DERNIER scan calculé : le rendu réutilise CE tensor pour tracer les
        # rayons (au lieu de rappeler read_scan → un 2e tirage de bruit). Ça (a) affiche
        # exactement ce que la policy a vu, et (b) garde le rollout DÉTERMINISTE à seed
        # fixé (aucune consommation RNG supplémentaire côté rendu) → indispensable pour
        # repérer une traversée réussie puis la rejouer à l'identique (finale).
        self._last_scan = out
        return out

    # ------------------------------------------------------------- interface
    def get_observations(self):
        if not getattr(self, "_ready", False):
            return TensorDict({"policy": torch.zeros(self.num_envs, 80, device=self.device)},
                              batch_size=[self.num_envs])
        pos, yaw = self._base_xy_yaw()
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        to_g = self.goal - pos
        rel_g = torch.stack([cy * to_g[:, 0] + sy * to_g[:, 1],
                             -sy * to_g[:, 0] + cy * to_g[:, 1]], dim=1) / 5.0   # but en repère robot
        obs = torch.cat([
            self.read_scan() / LIDAR_MAX,               # 72 — perception (obstacles)
            rel_g,                                       # 2  — but (direction + distance)
            self.lo_env.base_lin_vel[:, :2] * 0.5,       # 2
            self.lo_env.base_ang_vel[:, 2:3] * 0.25,     # 1
            self.last_nav_action,                        # 3
        ], dim=1)                                        # total : 80
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def reset(self):
        self._reset_idx(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.get_observations()

    def _reset_idx(self, mask):
        if not bool(mask.any()):
            return
        self.lo_env._reset_idx(mask)
        n = int(mask.sum())
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        # but droit devant (x=GOAL_X), décalage latéral aléatoire
        self.goal[idx, 0] = GOAL_X
        self.goal[idx, 1] = (torch.rand(n, device=self.device) * 2 - 1) * GOAL_Y_RANGE
        # obstacles répartis le long du couloir : x étagé + jitter, y aléatoire RECENTRÉ
        # (|y| ≤ 1,2 : passage garanti ≥ ~0,85 m entre l'obstacle et le mur, v3). Les
        # poteaux de BARRIÈRE et de MUR sont placés à part (ci-dessous).
        n_free = N_OBST - len(self._fence_ids) - len(self._wall_ids)
        for k in range(N_OBST):
            if OBSTACLES[k].get("fence") or OBSTACLES[k].get("wall"):
                continue
            gx = 1.8 + (GOAL_X - 2.6) * (k + 0.5) / n_free         # étagement régulier
            self.obst_home[k, idx, 0] = gx + (torch.rand(n, device=self.device) * 2 - 1) * 0.4
            self.obst_home[k, idx, 1] = (torch.rand(n, device=self.device) * 2 - 1) * 1.2
            self.obst_phase[k, idx] = torch.rand(n, device=self.device) * 2 * math.pi
            # piétons imprévisibles : fréquence, amplitude et cap propres à l'épisode ;
            # amplitude ÉCRÊTÉE pour que l'oscillation reste dans le couloir (pas de
            # piéton qui traverse le mur).
            self.obst_freq[k, idx] = 0.08 + torch.rand(n, device=self.device) * 0.10
            amp = 0.8 + torch.rand(n, device=self.device) * (MOVE_AMP - 0.8)
            amp_max = (WALL_Y - 0.5 - self.obst_home[k, idx, 1].abs()).clip(min=0.3)
            self.obst_amp[k, idx] = torch.minimum(amp, amp_max)
            ang = torch.rand(n, device=self.device) * math.pi
            self.obst_dir[k, idx, 0] = torch.cos(ang)
            self.obst_dir[k, idx, 1] = torch.sin(ang)
            if self.scenario in ("fixes", "goulot"):      # scènes vidéo sans mobiles
                self.obst_amp[k, idx] = 0.0
        # BARRIÈRE à PASSAGE ÉTROIT : rangée de poteaux en travers du couloir à
        # x aléatoire ; une « porte » de 1-2 poteaux (garés hors scène à x=60+) à
        # y aléatoire — le robot DOIT trouver et franchir le goulot (~1,3-1,9 m).
        # Scénarios "fixes"/"pietons" : barrière entièrement garée (pas de mur).
        xf = 3.8 + torch.rand(n, device=self.device) * 2.4
        y_gap = (torch.rand(n, device=self.device) * 2 - 1) * 1.5
        for j, k in enumerate(self._fence_ids):
            gy = FENCE_Y[j]
            parked = (y_gap - gy).abs() < FENCE_GAP
            if self.scenario in ("fixes", "pietons"):
                parked = torch.ones_like(parked)
            self.obst_home[k, idx, 0] = torch.where(parked, torch.full_like(xf, 60.0 + 2.0 * j), xf)
            self.obst_home[k, idx, 1] = torch.where(parked, torch.full_like(xf, 60.0),
                                                    torch.full_like(xf, gy))
            self.obst_phase[k, idx] = 0.0
        # MURS latéraux (v3) : palissade de poteaux serrés à y = ±WALL_Y sur toute la
        # longueur du couloir — présents dans TOUS les scénarios (le contournement par
        # l'extérieur devient physiquement impossible et vu au LiDAR).
        for j, k in enumerate(self._wall_ids):
            side = -1.0 if j < N_WALL else 1.0
            self.obst_home[k, idx, 0] = WALL_X0 + WALL_DX * (j % N_WALL)
            self.obst_home[k, idx, 1] = side * WALL_Y
            self.obst_phase[k, idx] = 0.0
        self._move_obstacles(moving_only=False)    # placer TOUS les obstacles (fixes + mobiles)
        pos, _ = self._base_xy_yaw()
        self.prev_dist[idx] = torch.linalg.norm(self.goal - pos, dim=1)[idx]
        self.last_nav_action[idx] = 0.0
        self.episode_length_buf[idx] = 0

    def _safety_filter(self, cmd):
        """COUCHE A — arrêt d'urgence DÉTERMINISTE (le garant, pas d'IA).
        Réimplémentation robot nu de la couche validée le 20/07 : d = min(scan
        frontal ±45°), seuil ADAPTATIF à la vitesse stop_d = 0.12 + 0.30·v (une
        garantie « jamais de contact » ne peut pas venir d'un réseau). Dans la
        bande [stop_d, stop_d+0.35] on RALENTIT linéairement ; sous stop_d,
        interdiction d'avancer (vx ≤ 0 — vy/yaw restent libres pour se dégager).
        La couche B (policy nav apprise) propose, la couche A dispose."""
        scan = self.read_scan()
        d_front = torch.where(self._front_mask.unsqueeze(0), scan,
                              torch.full_like(scan, LIDAR_MAX)).min(dim=1).values
        v = self.lo_env.base_lin_vel[:, 0].clip(min=0.0)
        stop_d = 0.12 + 0.30 * v
        factor = ((d_front - stop_d) / 0.35).clip(0.0, 1.0)
        vx = cmd[:, 0]
        cmd[:, 0] = torch.where(vx > 0.0, vx * factor, vx)
        return cmd

    def step(self, nav_actions):
        nav_actions = torch.clip(nav_actions, -1.0, 1.0)
        cmd = self.cmd_low + (nav_actions + 1.0) * 0.5 * (self.cmd_high - self.cmd_low)
        if self.safety_stop:
            cmd = self._safety_filter(cmd)

        fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf += 1
        self._move_obstacles()                     # avance les piétons AVANT la sous-boucle
        with torch.no_grad():
            for _ in range(NAV_DECIM):
                self.lo_env.commands.copy_(cmd)
                self.lo_env._update_observation()
                lo_act = self.actor_lo(self.lo_env.obs_buf)
                self.lo_env.step(lo_act)
                fell |= self.lo_env.reset_buf
        pos, yaw = self._base_xy_yaw()
        to_g = self.goal - pos
        dist = torch.linalg.norm(to_g, dim=1)
        bearing = torch.atan2(to_g[:, 1], to_g[:, 0])
        hdg_err = torch.remainder(bearing - yaw + math.pi, 2 * math.pi) - math.pi
        clear = self._clearance(pos)

        # ---- récompenses
        prog = (self.prev_dist - dist).clip(-0.4, 0.4)
        r_progress = prog * 12.0                                    # avancer vers le but
        r_heading = torch.cos(hdg_err) * 0.05                       # viser le but
        # couche B : pénalité de proximité qui croît quand la garde < 0.6 m (borne douce)
        prox = (0.6 - clear).clip(min=0.0)
        r_clearance = -torch.square(prox) * 6.0
        collided = clear < COLLIDE_MARGIN
        r_collision = torch.where(collided, torch.full_like(dist, -12.0), torch.zeros_like(dist))
        r_fall = torch.where(fell, torch.full_like(dist, -10.0), torch.zeros_like(dist))
        # CONTOURNEMENT (v3) : sortir du couloir (au-delà des murs, ou repartir loin en
        # arrière) = mission ratée, pénalisée comme une collision — l'évitement doit se
        # faire ENTRE les obstacles, pas autour du parcours.
        escaped = (pos[:, 1].abs() > WALL_Y + 0.2) | (pos[:, 0] < -2.0)
        r_escape = torch.where(escaped, torch.full_like(dist, -12.0), torch.zeros_like(dist))
        success = dist < ARRIVE
        r_success = torch.where(success, torch.full_like(dist, 25.0), torch.zeros_like(dist))
        r_time = torch.full_like(dist, -0.02)
        r_action = -0.05 * torch.sum((nav_actions - self.last_nav_action) ** 2, dim=1)
        self.rew_buf = (r_progress + r_heading + r_clearance + r_collision + r_fall
                        + r_escape + r_success + r_time + r_action)
        for key, v in [("progress", r_progress), ("clearance", r_clearance),
                       ("collision", r_collision), ("success", r_success), ("fall", r_fall),
                       ("heading", r_heading), ("action", r_action), ("escape", r_escape)]:
            self.episode_sums[key] += v

        # ---- terminaisons
        timeout = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = success | fell | collided | escaped | timeout
        self.extras["time_outs"] = timeout.to(gs.tc_float)

        self.extras["episode"] = {}
        if bool(self.reset_buf.any()):
            m = self.reset_buf
            nn = m.sum()
            self.extras["episode"]["n_success"] = success[m].to(gs.tc_float).sum() / nn
            for key, v in self.episode_sums.items():
                self.extras["episode"][f"rew_{key}"] = v[m].sum() / nn
                v[m] = 0.0
            self._reset_idx(m)

        self.prev_dist = torch.where(self.reset_buf, self.prev_dist, dist)
        self.last_nav_action = torch.where(self.reset_buf.unsqueeze(1),
                                           torch.zeros_like(nav_actions), nav_actions)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras
