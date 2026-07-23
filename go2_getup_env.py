"""SE RELEVER (get-up) — le robot part COUCHÉ en orientation aléatoire et se remet debout.

Sous-classe simple de Go2Env (comme le terrain). Différences vs marche :
  - AUCUNE terminaison sur tangage/roulis (sinon l'épisode meurt dès qu'il est au sol) :
    seul le timeout termine → le robot a le temps de se relever. (à régler côté train :
    termination_if_pitch/roll_greater_than très grand).
  - init : quaternion UNIFORME aléatoire (flanc / dos / en vrac), hauteur basse.
  - récompense = SE REDRESSER (gravité projetée z → -1), atteindre la hauteur debout,
    poser les 4 pieds, puis tenir immobile (commande nulle).

Warm-start depuis le marcheur (il connaît déjà la posture debout) : le get-up n'est
« que » la remontée depuis le sol. C'est le plus exploratoire des postures.
"""
import torch

import genesis as gs

from go2_env import Go2Env


class Go2GetupEnv(Go2Env):
    def __init__(self, *args, **kwargs):
        self._getup_ready = False
        super().__init__(*args, **kwargs)
        self._getup_ready = True
        self.reset()

    def _reset_idx(self, envs_idx=None):
        super()._reset_idx(envs_idx)
        if not getattr(self, "_getup_ready", False):
            return
        mask = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                if envs_idx is None else envs_idx)
        n = int(mask.sum())
        if n == 0:
            return
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        q = torch.zeros((n, 7 + self.num_actions), dtype=gs.tc_float, device=self.device)
        q[:, 2] = 0.18                                   # près du sol (couché)
        rq = torch.randn(n, 4, device=self.device)       # orientation uniforme aléatoire
        q[:, 3:7] = rq / torch.linalg.norm(rq, dim=1, keepdim=True)
        base_ang = self.init_dof_pos.unsqueeze(0).repeat(n, 1)
        q[:, 7:] = base_ang + (torch.rand(n, self.num_actions, device=self.device) * 2 - 1) * 0.3
        self.robot.set_qpos(q, envs_idx=idx, zero_velocity=True, skip_forward=True)
        self.base_pos[idx] = q[:, :3]

    # ---------- récompenses get-up (bornées) ----------
    def _reward_upright(self):
        # gravité projetée z ≈ -1 quand DEBOUT, +1 sur le dos → on récompense le redressement.
        return torch.nan_to_num((-self.projected_gravity[:, 2]).clip(-1.0, 1.0), nan=0.0)

    def _reward_feet_down(self):
        f = self.robot.get_links_net_contact_force()
        contact = torch.linalg.norm(f[:, self.feet_idx], dim=2) > 1.0
        return torch.nan_to_num(contact.float().sum(dim=1) / 4.0, nan=0.0)

    def _reward_base_height(self):
        # erreur de hauteur bornée (au sol l'erreur est grande → pousse à monter/tenir debout)
        err = (self.base_pos[:, 2] - self.reward_cfg["base_height_target"]).clip(-0.5, 0.5)
        return torch.nan_to_num(torch.square(err), nan=0.25, posinf=0.25, neginf=0.25)

    # ---------- propreté du mouvement (demande 22/07 : relevé « catastrophe ») ----------
    def _reward_dof_vel_soft(self):
        # anti-GESTICULATION : pénalité DOUCE des vitesses articulaires pendant tout
        # l'épisode. Assez faible pour laisser la poussée vigoureuse du relevé (leçon
        # 21/07 : une pénalité de mouvement forte tue la transition), assez présente
        # pour préférer un relevé fluide au battage de pattes.
        v2 = torch.sum(torch.square(self.dof_vel), dim=1).clip(max=400.0)
        return torch.nan_to_num(v2, nan=0.0, posinf=400.0, neginf=0.0)

    def _reward_vertical_settle(self):
        # ANTI-REBOND (correctif 22/07, retour vidéo « le robot rebondit sur le sol ») :
        # pénalise la vitesse verticale de la base au CARRÉ, bornée. Un relevé FLUIDE a un
        # v_z modéré (~0,3-0,4 m/s → v_z²≈0,1 → coût quasi nul) ; un REBOND / une remontée
        # trop violente a des pics de v_z (>1 m/s → v_z²>1 → coût élevé). L'asymétrie fait
        # que le terme cible les impacts brusques SANS empêcher la remontée nécessaire.
        # NE touche AUCUNE limite physique (couple/vitesse/position restent enforce) — on
        # ne fait que PRÉFÉRER une trajectoire plus douce dans l'espace des solutions valides.
        vz2 = torch.square(self.base_lin_vel[:, 2]).clip(max=6.0)
        return torch.nan_to_num(vz2, nan=0.0, posinf=6.0, neginf=0.0)

    def _reward_hold_upright(self):
        # immobilité STRICTE conditionnée à l'ÉTAT (pas au temps) : dès que le robot
        # est debout (gravité projetée ≈ -z ET hauteur debout), pattes et base doivent
        # se figer — le relevé se termine par un arrêt net, sans piétinement.
        up = (self.projected_gravity[:, 2] < -0.85) & (self.base_pos[:, 2] > 0.26)
        joints = torch.sum(torch.square(self.dof_vel), dim=1).clip(0.0, 60.0)
        base = torch.sum(torch.square(self.base_lin_vel), dim=1).clip(0.0, 12.0)
        return torch.nan_to_num(up.float() * (0.2 * joints + 3.0 * base), nan=0.0)
