"""LOCOMOTION TOUT-TERRAIN — robot NU (segment « franchir obstacles »).

Le Go2 (sans caméra ni bras) apprend à marcher sur TOUS types de surfaces (relief
type gravier, pentes, vagues, obstacles discrets, marches, escaliers pyramidaux)
au lieu du sol plat. Warm-start depuis le marcheur `go2-walking` : il sait déjà
marcher droit, il « n'a qu'à » encaisser le relief et le franchir.

Choix qui maximisent réussite ET réalisme sim→réel :
  - OBS PROPRIOCEPTIVE PURE (45-D, identique au marcheur) : le robot réagit à ce
    qu'il RESSENT (attitude, vitesses, positions articulaires), PAS à une carte de
    hauteur parfaite qu'il n'aurait jamais sur le vrai robot. Bonus : même obs →
    warm-start direct depuis le marcheur.
  - HAUTEUR DE BASE mesurée RELATIVEMENT au terrain sous le robot (sinon la pénalité
    de hauteur se bat contre le relief → le robot se fige accroupi ou est éjecté).
  - SPAWN réparti sur toutes les tuiles (faciles → dures), à la BONNE altitude via
    interpolation bilinéaire du champ de hauteur (le plus-proche-voisin se trompe de
    pente×pas et fait naître le robot enfoncé/flottant → éjection immédiate).
  - RÉALISME hérité (couples/vitesses Unitree bornés + friction/poussées/charge/bruit
    capteurs randomisés) via `realism.apply_realism`, appliqué côté script d'entraînement.
  - ARRÊT MAÎTRISÉ SUR LE RELIEF : `stand_still` + `dof_freeze_stop` (hérités) restent
    actifs → le robot doit s'immobiliser proprement même sur une pente (demandé 21/07).

Presque tout est déjà dans `Go2Env` (orientation, feet_air_time, undesired_contact,
stand_still, dof_freeze_stop, dof_vel_limit, DR, bruit). Cette sous-classe n'ajoute
que le TERRAIN : géométrie, spawn en relief, et hauteur de base relative.
"""
import numpy as np
import torch

import genesis as gs

from go2_env import Go2Env


class Go2TerrainEnv(Go2Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # géométrie du terrain (pour spawn à la bonne altitude sur chaque tuile)
        tc = self.env_cfg["terrain_cfg"]
        self.h_scale = tc["horizontal_scale"]
        self.v_scale = tc["vertical_scale"]
        self.n_rows, self.n_cols = tc["n_subterrains"]
        self.sub_m = tc["subterrain_size"][0]        # tuile carrée (m)
        hf = np.asarray(self.terrain.terrain_hf, dtype=np.float32)
        self.hf = torch.tensor(hf, device=self.device)      # (Hx, Hy) indices entiers
        self.hf_nx, self.hf_ny = self.hf.shape
        self.spawn_margin = 1.5                       # ne pas spawner au bord d'une tuile
        self._terrain_ready = True
        self.reset()                                  # respawn EN RELIEF (le reset du parent
        #                                               a spawné à plat, terrain pas encore prêt)

    # ---------- hauteur du terrain à une position monde (x, y), bilinéaire ----------
    def _terrain_h(self, xy):
        fx = xy[:, 0] / self.h_scale
        fy = xy[:, 1] / self.h_scale
        x0 = fx.floor().long().clamp(0, self.hf_nx - 2)
        y0 = fy.floor().long().clamp(0, self.hf_ny - 2)
        x1, y1 = x0 + 1, y0 + 1
        tx = (fx - x0.float()).clamp(0.0, 1.0)
        ty = (fy - y0.float()).clamp(0.0, 1.0)
        h = (self.hf[x0, y0] * (1 - tx) * (1 - ty)
             + self.hf[x1, y0] * tx * (1 - ty)
             + self.hf[x0, y1] * (1 - tx) * ty
             + self.hf[x1, y1] * tx * ty)
        return h * self.v_scale

    # ---------- reset : repositionne sur une tuile aléatoire, à la bonne altitude ----------
    def _reset_idx(self, envs_idx=None):
        # parent : buffers, friction/charge randomisées, resample commandes, set_qpos plat
        super()._reset_idx(envs_idx)
        if not getattr(self, "_terrain_ready", False):
            return
        if envs_idx is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = envs_idx
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        n = int(idx.numel())
        if n == 0:
            return
        # tuile aléatoire (ligne = difficulté, colonne = variante) + offset central
        row = torch.randint(0, self.n_rows, (n,), device=self.device)
        col = torch.randint(0, self.n_cols, (n,), device=self.device)
        off = (torch.rand(n, 2, device=self.device) * 2 - 1) * (self.sub_m / 2 - self.spawn_margin)
        cx = (row.float() + 0.5) * self.sub_m + off[:, 0]
        cy = (col.float() + 0.5) * self.sub_m + off[:, 1]
        xy = torch.stack([cx, cy], dim=1)
        z = self._terrain_h(xy) + self.env_cfg["base_init_pos"][2]
        # qpos par env : base (xyz + quat neutre) + angles de repos des pattes
        q = torch.zeros((n, 7 + self.num_actions), dtype=gs.tc_float, device=self.device)
        q[:, 0], q[:, 1], q[:, 2] = cx, cy, z
        q[:, 3] = 1.0                                 # quat w=1 (orientation neutre)
        q[:, 7:] = self.init_dof_pos.unsqueeze(0)
        self.robot.set_qpos(q, envs_idx=idx, zero_velocity=True, skip_forward=True)
        self.base_pos[idx] = q[:, :3]
        if self.feet_air_time is not None:
            self.feet_air_time[idx] = 0.0

    # ---------- hauteur de base RELATIVE au terrain (au lieu de l'absolue du parent) ----------
    def _reward_base_height(self):
        # le -base_height absolu du parent se battrait contre le relief ; ici on mesure
        # la garde au sol RÉELLE sous le robot. Erreur bornée ±0.5 m + nan_to_num : sur
        # le relief un état physique aberrant (pénétration) produirait un carré géant qui
        # empoisonnerait le gradient PPO.
        rel = self.base_pos[:, 2] - self._terrain_h(self.base_pos[:, :2])
        err = (rel - self.reward_cfg["base_height_target"]).clip(-0.5, 0.5)
        return torch.nan_to_num(torch.square(err), nan=0.25, posinf=0.25, neginf=0.25)
