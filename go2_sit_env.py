"""S'ASSEOIR — le robot part DEBOUT et prend la posture assise (arrière-train au sol,
avant relevé), de façon CONTRÔLÉE et PROGRESSIVE (pas d'un coup). Sous-classe de Go2Env.

Approche v3 (22/07, après échec de l'assise phasée) : le suivi de trajectoire temporelle
était trop dur à apprendre — PPO restait DEBOUT. On revient à des CIBLES STATIQUES
(assiette + hauteur d'arrière-train) et on LIMITE LA VITESSE VERTICALE du corps
(`lin_vel_z` dans le train) : la posture assise est atteinte à coup sûr, mais LENTEMENT.

Récompense d'ASSIETTE via la GRAVITÉ PROJETÉE (pas les angles d'Euler) :
nez relevé de θ ⇔ projected_gravity_x = -sin(θ). Formulation indépendante de la
convention de signe (audit 21/07 : quat_to_xyz Genesis donne un tangage NÉGATIF
nez en l'air) et sans singularité de cardan. Départ debout, commande nulle,
warm-start marcheur, pas de terminaison tangage. Réalisme hérité.
"""
import math

import torch

import genesis as gs

from go2_env import Go2Env


class Go2SitEnv(Go2Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    # ---------- récompenses assise (bornées, cibles STATIQUES) ----------
    def _reward_sit_pitch(self):
        # tronc incliné vers l'arrière (NEZ RELEVÉ de θ degrés) : gravité projetée
        # sur l'axe X du corps proche de -sin(θ). Gaussienne bornée [0,1].
        # LARGEUR σ=0,6 (et non 0,25) — correctif 22/07 « il ne s'assoit jamais » : à
        # σ=0,25, depuis DEBOUT (pg_x≈0, err≈0,64) la gaussienne vaut ~0,001 → PLATE →
        # aucun gradient pour AMORCER le cabrage → PPO reste debout (optimum local, même
        # piège que le coucher phasé). À σ=0,6 elle vaut ~0,32 debout et croît dès qu'il
        # se cabre : le gradient existe tout du long, il apprend à s'asseoir.
        target = math.sin(math.radians(self.reward_cfg.get("sit_pitch_target", 40.0)))
        err = self.projected_gravity[:, 0] + target        # cible : pg_x = -sin(θ)
        # COARSE+FINE (correctif 23/07, « il s'étale vers l'avant au lieu de s'asseoir ») :
        # à σ=0,6 SEUL, un cabrage PARTIEL (~20°) valait déjà ~0,78 et l'assis COMPLET ~1,0
        # → +0,2 seulement, insuffisant pour justifier l'effort/instabilité → PPO se figeait
        # à mi-cabrage (étalement bas). COARSE (σ=0,6) garde le gradient d'AMORÇAGE depuis
        # debout (sinon plat, cf. σ=0,25) ; FINE (σ=0,18) récompense FORTEMENT l'aboutissement
        # (partiel→complet passe de ~0,35 à 1,0) → le VRAI assis domine nettement le demi-assis.
        coarse = torch.exp(-torch.square(err / 0.6))
        fine = torch.exp(-torch.square(err / 0.18))
        return torch.nan_to_num(0.4 * coarse + 0.6 * fine, nan=0.0)

    def _reward_rear_low(self):
        # arrière-train bas : hauteur de base MODÉRÉE (entre debout 0,38 et couché 0,12).
        target = self.reward_cfg.get("sit_height_target", 0.26)
        err = (self.base_pos[:, 2] - target).clip(-0.4, 0.4)
        return torch.nan_to_num(torch.square(err), nan=0.16, posinf=0.16, neginf=0.16)

    def _reward_front_tuck(self):
        # PIEDS AVANT RANGÉS SOUS LE CORPS (correctif 23/07 v3, « assise avachie ») : la v2
        # atteignait nez-haut + train-bas mais en se calant sur des pattes avant ÉTALÉES vers
        # l'avant (rien dans la récompense ne contrôlait la position des pieds) → assis penché.
        # On récompense une faible distance HORIZONTALE pied-avant ↔ centre du corps (xy monde,
        # donc indépendante de l'inclinaison nez-haut). Assis PROPRE : pieds avant ~0,22 m sous
        # les épaules → récompense ~1 ; étalés vers l'avant (>0,38 m) → récompense ~0,17.
        # NB : debout donne aussi des pieds rangés, mais debout perd sit_pitch (nez) + rear_low
        # (hauteur) → l'assise reste l'optimum ; ce terme n'écarte QUE la variante étalée.
        pos = self.robot.get_links_pos()                     # (E, L, 3) monde
        base_xy = self.base_pos[:, :2]
        fl = torch.linalg.norm(pos[:, self.feet_idx[0], :2] - base_xy, dim=1)
        fr = torch.linalg.norm(pos[:, self.feet_idx[1], :2] - base_xy, dim=1)
        d = 0.5 * (fl + fr)
        return torch.nan_to_num(torch.exp(-torch.square((d - 0.22) / 0.12)), nan=0.0)

    def _reward_rear_tuck(self):
        # PATTES ARRIÈRE REPLIÉES SOUS LA CROUPE (correctif v4, 23/07). Diagnostic corrigé :
        # ce n'est pas l'avant mais l'ARRIÈRE qui s'étale — les pieds arrière partent vers
        # l'avant au lieu de se ranger sous les hanches (assis « écartelé » derrière).
        # ATTENTION : la distance au CENTRE du corps ne discrimine pas l'arrière (un pied
        # avancé se RAPPROCHE du centre). On mesure donc la distance HORIZONTALE pied-arrière
        # ↔ ATTACHE de sa patte (lien `*_thigh`) : replié → pied ~sous la hanche (petite
        # distance) ; étendu devant → grande distance. Récompense bornée [0,1], insensible
        # à l'inclinaison (xy monde). Pur reward shaping — `get_links_pos()` lit la position
        # post-contact, physique/limites/self-collision intactes.
        pos = self.robot.get_links_pos()                     # (E, L, 3) monde
        if not hasattr(self, "_rear_pairs"):
            names = [l.name for l in self.robot.links]

            def _find(cands):
                for c in cands:
                    if c in names:
                        return names.index(c)
                return None
            rl = _find(["RL_thigh", "RL_hip"])
            rr = _find(["RR_thigh", "RR_hip"])
            self._rear_pairs = ([(self.feet_idx[2], rl), (self.feet_idx[3], rr)]
                                if rl is not None and rr is not None else None)
        if not self._rear_pairs:
            return torch.zeros(self.num_envs, device=self.device)
        d = 0.0
        for foot_i, hip_i in self._rear_pairs:
            d = d + torch.linalg.norm(pos[:, foot_i, :2] - pos[:, hip_i, :2], dim=1)
        d = d / len(self._rear_pairs)                         # replié ~0,12 m ; étendu >0,35 m
        return torch.nan_to_num(torch.exp(-torch.square(d / 0.18)), nan=0.0)
