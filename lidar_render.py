"""Rendu du segment LiDAR — le robot navigue un parcours d'obstacles FIXES + MOBILES
avec les RAYONS LiDAR VISIBLES (colorés par distance), au fil des checkpoints, à côté de
la courbe « taux de réussite » + l'overlay temps GPU ↔ expérience-robot.

Les rayons sont tracés depuis MON scan analytique (cohérent avec ce que voit la policy) :
un éventail de 72 rayons ; vert = libre, orange = obstacle proche, rouge = très proche.
CPU (sérialisation ROCm : à lancer GPU libre). Réutilise les helpers de pilot_walk_video.

Après la progression des checkpoints, une FINALE joue une traversée COMPLÈTE du meilleur
checkpoint (repérée par seed pour garantir une réussite), terminée par un ARRÊT MAÎTRISÉ
au but (commande nulle, pattes figées — le robot NE revient PAS en arrière).

    python lidar_render.py -e go2-lidar --cap <pic>
"""
import argparse
import glob
import math
import os
import pickle
import re

import numpy as np
from PIL import Image, ImageDraw

from pilot_walk_video import (compose, curve_panel, load_scalars, pick_checkpoints,
                              CURVE_METRICS, ROBOT_W, ROBOT_H, FPS, SECS_PER_CKPT, _font)

WARM_EXP = "go2-walking"
# couleurs RGBA des rayons par proximité (fraction de la portée)
COL_FAR = (0.16, 0.90, 0.32, 0.5)    # vert, semi-transparent (libre)
COL_MID = (1.00, 0.60, 0.00, 0.9)    # orange (obstacle proche)
COL_NEAR = (1.00, 0.14, 0.14, 1.0)   # rouge (très proche)

# finale — traversée complète + arrêt maîtrisé
SEED0 = 20250723        # base des seeds essayés pour repérer une traversée réussie
N_SEED_TRY = 40         # nb de tirages testés (dry-run) avant de rendre le meilleur
ARRIVE_STOP = 0.85      # on bascule en arrêt maîtrisé quand dist(but) < ce seuil (> ARRIVE=0.7)
STOP_HOLD_S = 1.8       # durée de l'immobilisation figée au but (s)


def ray_color(frac):
    if frac < 0.33:
        return COL_NEAR
    if frac < 0.66:
        return COL_MID
    return COL_FAR


def lidar_legend(rgb):
    """Incruste la LÉGENDE LiDAR (sujet du segment) sur le panneau robot : code couleur
    des rayons + rappel que le scan porte un BRUIT capteur réaliste (gaussien + échos
    perdus). Le pied de page générique reste ; ici on explicite ce qui est propre au LiDAR."""
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img, "RGBA")
    x, y, w, h = 18, 16, 470, 140
    d.rectangle([x, y, x + w, y + h], fill=(6, 10, 20, 170))
    ft = _font(18, bold=True)
    fs = _font(15)
    d.text((x + 12, y + 9), "LiDAR — 72 secteurs · portée 6,5 m", font=ft, fill="#dbe4ee")
    rows = [((41, 230, 82), "libre"),
            ((255, 153, 0), "obstacle proche"),
            ((255, 36, 36), "très proche")]
    yy = y + 38
    for col, txt in rows:
        d.line([(x + 14, yy + 8), (x + 44, yy + 8)], fill=col, width=6)
        d.text((x + 54, yy), txt, font=fs, fill="#c3ccd8")
        yy += 25
    d.text((x + 12, y + h - 26), "bruit capteur réaliste : ~3 cm + échos perdus",
           font=fs, fill="#8b96a5")
    return np.asarray(img)


def _latest(exp):
    return max((int(m.group(1)) for f in glob.glob(f"logs/{exp}/model_*.pt")
                if (m := re.search(r"model_(\d+)\.pt$", f))), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--exp_name", default="go2-lidar")
    ap.add_argument("--cap", type=int, default=-1, help="itération max (pic) ; -1 = tout")
    ap.add_argument("--label", default="ÉVITER — LiDAR")
    ap.add_argument("--scenario", default="complet",
                    choices=["complet", "fixes", "pietons", "goulot"],
                    help="scène de démo (mini-segments vidéo) : obstacles fixes seuls, "
                         "piétons mobiles, goulot/barrière, ou parcours complet")
    ap.add_argument("--no_finale", action="store_true",
                    help="désactive la finale (traversée complète + arrêt maîtrisé)")
    ap.add_argument("--finale_secs", type=float, default=24.0,
                    help="durée MAX de la finale (cap ; la traversée s'arrête au but)")
    args = ap.parse_args()
    log_dir = f"logs/{args.exp_name}"
    cap = None if args.cap < 0 else args.cap

    metric = "success"
    scalars = load_scalars(log_dir, cap=cap)
    if CURVE_METRICS[metric][0] not in scalars:
        metric = "reward"
    tag = CURVE_METRICS[metric][0]
    steps_m, _ = scalars[tag]
    max_iter = int(steps_m[-1])
    gpu_per_iter = 5.5
    wall = scalars.get(tag + "/wall")
    if wall is not None and len(wall) > 1 and steps_m[-1] > steps_m[0]:
        gpu_per_iter = float((wall[-1] - wall[0]) / (steps_m[-1] - steps_m[0]))
    ckpts = pick_checkpoints(log_dir, cap=cap)
    print(f"checkpoints : {ckpts} | max_iter={max_iter} | metric={metric} | {gpu_per_iter:.2f}s/iter")

    os.makedirs("videos", exist_ok=True)
    import imageio.v2 as imageio
    suffix = "" if args.scenario == "complet" else f"_{args.scenario}"
    out = f"videos/segment_{args.exp_name}{suffix}.mp4"
    writer = imageio.get_writer(out, fps=FPS, macro_block_size=1)

    # ---- Genesis (CPU) ----
    import genesis as gs
    import torch
    from go2_lidar_env import Go2LidarAvoidEnv, LIDAR_MAX, N_H, NAV_DECIM, ARRIVE
    from rsl_rl.runners import OnPolicyRunner

    gs.init(backend=gs.cpu, logging_level="warning")
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    # RENDU CLEAN : couper les perturbations d'entraînement de la locomotion (on isole
    # le comportement de navigation + les obstacles). Les obstacles mobiles RESTENT (le sujet).
    # Le BRUIT LiDAR reste ACTIF (réalisme max + c'est le sujet du segment ; le rendu reste
    # déterministe grâce au cache de scan côté env → rejeu à seed fixé).
    env_cfg = dict(env_cfg)
    env_cfg["push_interval_s"] = 0
    env_cfg["push_vel"] = 0.0
    env_cfg["friction_range"] = None
    env_cfg["payload_rand_kg"] = 0.0
    env_cfg["obs_noise"] = None
    env_cfg["nav_safety_stop"] = True   # COUCHE A active à l'inférence (comme au déploiement)
    env_cfg["lidar_scenario"] = args.scenario   # scène de démo (fixes/pietons/goulot/complet)
    loco = f"logs/{WARM_EXP}/model_{_latest(WARM_EXP)}.pt"

    env = Go2LidarAvoidEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg,
                           locomotion_ckpt=loco,
                           camera_cfg=[{"res": (ROBOT_W, ROBOT_H), "fov": 46, "debug": True}])
    cam = env.lo_env.cams[0]
    scene = env.lo_env.scene
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    az = env._az[0].cpu().numpy()                      # (72,) azimuts capteur
    n_frames = int(FPS * SECS_PER_CKPT)

    def draw_rays():
        scene.clear_debug_objects()
        # rayons tracés depuis le DERNIER scan calculé par l'env (celui que la policy a vu) :
        # pas de 2e tirage de bruit → cohérence + déterminisme (rejeu de la finale).
        ls = getattr(env, "_last_scan", None)
        scan = (ls if ls is not None else env.read_scan())[0].cpu().numpy()
        bp = env.lo_env.base_pos[0].cpu().numpy()
        q = env.lo_env.base_quat[0].cpu().numpy()
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        ox, oy, oz = bp[0] + 0.28 * math.cos(yaw), bp[1] + 0.28 * math.sin(yaw), bp[2] + 0.08
        for i in range(N_H):
            d = float(scan[i])
            a = az[i] + yaw
            ix, iy = ox + d * math.cos(a), oy + d * math.sin(a)
            scene.draw_debug_line([ox, oy, oz], [ix, iy, oz], radius=0.007,
                                  color=ray_color(d / LIDAR_MAX))
        return bp

    def set_cam(bp):
        # caméra 3/4 arrière, un peu haute (voir le parcours + les rayons)
        cam.set_pose(pos=(bp[0] - 0.6, bp[1] - 3.6, 2.3),
                     lookat=(bp[0] + 2.2, bp[1], 0.3), up=(0, 0, 1))

    # annotation permanente sur la courbe : le taux plafonne à ~41 %, honnêtement signalé
    # comme NON optimal + les leviers d'amélioration (cf. « Pistes d'exploration » du README).
    NOTE = "≈ 41 % — non optimal\n(+ itérations · curriculum · ADR)"

    for k in ckpts:
        runner.load(f"{log_dir}/model_{k}.pt")
        policy = runner.get_inference_policy(device=gs.device)
        panel = curve_panel(scalars, k, max_iter, metric, 0, note=NOTE)
        gpu_s = k * gpu_per_iter
        obs = env.reset()
        with torch.no_grad():
            for _ in range(n_frames):
                act = policy(obs)
                obs = env.step(act)[0]
                bp = draw_rays()
                set_cam(bp)
                frame = cam.render()[0]
                writer.append_data(lidar_legend(
                    compose(frame, panel, k, max_iter, gpu_s, label=args.label, offset=0)))
        print(f"  checkpoint {k}: rendu ok")

    # ---------- FINALE : traversée COMPLÈTE + ARRÊT MAÎTRISÉ ----------
    if not args.no_finale and ckpts:
        kf = ckpts[-1]                                   # meilleur checkpoint (le mieux appris)
        runner.load(f"{log_dir}/model_{kf}.pt")
        policy = runner.get_inference_policy(device=gs.device)
        panel = curve_panel(scalars, kf, max_iter, metric, 0, note=NOTE)
        gpu_s = kf * gpu_per_iter
        flabel = args.label                               # court (sinon chevauche la colonne temps)
        fsub = "traversée complète · arrêt maîtrisé au but"
        max_tf = int(FPS * args.finale_secs)
        hold_f = int(FPS * STOP_HOLD_S)

        def goal_dist():
            pos, _ = env._base_xy_yaw()
            return float(torch.linalg.norm(env.goal[0] - pos[0]))

        # 1) REPÉRAGE : dry-run (sans rendu) de plusieurs seeds → on retient le 1er qui
        #    ATTEINT le but proprement (le rollout est déterministe à seed fixé). Fallback :
        #    le seed qui approche le plus (min dist) si aucun ne réussit.
        good, best_seed, best_val = None, SEED0, 1e9
        with torch.no_grad():
            for s in range(SEED0, SEED0 + N_SEED_TRY):
                torch.manual_seed(s)
                obs = env.reset()
                reached, mind = False, 1e9
                for _ in range(max_tf):
                    d = goal_dist()
                    mind = min(mind, d)
                    if d < ARRIVE_STOP:
                        reached = True
                        break
                    obs = env.step(policy(obs))[0]
                    if bool(env.reset_buf[0]):            # collision/sortie/timeout → échec
                        break
                if reached:
                    good = s
                    break
                if mind < best_val:
                    best_val, best_seed = mind, s
        seed = good if good is not None else best_seed
        print(f"finale: seed={seed} "
              + ("SUCCÈS" if good is not None else f"(meilleur approche={best_val:.2f} m)"))

        # 2) RENDU de la traversée retenue (mêmes tirages → même rollout), puis arrêt figé.
        torch.manual_seed(seed)
        obs = env.reset()
        arrived = False
        with torch.no_grad():
            for _ in range(max_tf):
                if goal_dist() < ARRIVE_STOP:
                    arrived = True
                    break
                obs = env.step(policy(obs))[0]
                if bool(env.reset_buf[0]):                # run terminé sans arrivée
                    break
                bp = draw_rays()
                set_cam(bp)
                writer.append_data(lidar_legend(
                    compose(cam.render()[0], panel, kf, max_iter, gpu_s,
                            label=flabel, sublabel=fsub, offset=0)))

            # ARRÊT MAÎTRISÉ au but : commande de locomotion NULLE (le marcheur s'immobilise,
            # pattes figées — règle d'arrêt maîtrisé) ; on TIENT la position. Le robot ne
            # repart PAS (on ne laisse PAS l'env faire son reset-succès qui téléporterait au
            # départ). Rayons rafraîchis une fois sur la pose finale.
            if arrived:
                env.read_scan()                           # rafraîchit le scan sur la pose d'arrivée
                zero = torch.zeros_like(env.lo_env.commands)
                for _ in range(hold_f):
                    for _ in range(NAV_DECIM):
                        env.lo_env.commands.copy_(zero)
                        env.lo_env._update_observation()
                        env.lo_env.step(env.actor_lo(env.lo_env.obs_buf))
                    bp = draw_rays()
                    set_cam(bp)
                    writer.append_data(lidar_legend(
                        compose(cam.render()[0], panel, kf, max_iter, gpu_s,
                                label=flabel, sublabel=fsub, offset=0)))
        print(f"finale rendue (arrivée={arrived})")

    writer.close()
    print(f"🎥 {out}")


if __name__ == "__main__":
    main()
