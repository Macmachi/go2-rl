"""Vidéo composite « le robot apprend » — rendu d'un SEGMENT (marcher, courir, terrain…).

Assemble, côte à côte :
  • GAUCHE  : le Go2 nu rejoué AU CHECKPOINT courant (gigote → titube → maîtrise) ;
  • DROITE  : la courbe d'apprentissage TensorBoard (récompense OU durée-avant-chute)
              qui se remplit au même rythme, avec un marqueur sur l'itération courante ;
  • BANDEAU : équivalence « temps de calcul GPU ↔ expérience-robot » + itération.

Générique : `--label`, `--curve-metric` (reward|episode_length), `--iter-offset` (pour
un segment warm-starté dont les itérations absolues démarrent à 2500+ → axe 0-based),
`--cap`, `--vx`. Le rendu Genesis est CPU : à lancer UNIQUEMENT quand aucun entraînement
GPU ne tourne (sérialisation ROCm). La moitié « courbe » (--curve-only) se teste sans Genesis.

    # marche (pilote, from-scratch)
    python pilot_walk_video.py -e go2-walking
    # course (warm-startée à 2499, la durée-avant-chute raconte mieux que le reward)
    python pilot_walk_video.py -e go2-run --iter-offset 2500 --curve-metric episode_length \
        --cap -1 --label COURIR --vx 2.3
    python pilot_walk_video.py -e go2-walking --curve-only    # test courbe (sans Genesis)
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ---- géométrie du composite (dimensions paires pour l'encodage vidéo) ----
ROBOT_W, ROBOT_H = 960, 720
CURVE_W = 640
BANNER_H = 152
CANVAS_W = ROBOT_W + CURVE_W          # 1600
CANVAS_H = ROBOT_H + BANNER_H         # 872

# légende réalisme (petit texte permanent en bas). « Entraîné sous » : le rendu isole
# le comportement appris (perturbations coupées), mais l'entraînement, lui, les subit.
REALISM_CAPTION = ("Entraîné en physique réaliste — couple 23,7/35,55 N·m & vitesses 30,1/20,07 rad/s bornés "
                   "(Unitree Go2) · friction, poussées, charge & bruit capteurs randomisés")
FPS = 50
SECS_PER_CKPT = 3.0                   # durée de chaque clip de checkpoint

# ---- constantes d'expérience (cf. go2_env / go2_train) ----
NUM_ENVS = 4096
STEPS_PER_ITER = 24
DT = 0.02
EXP_PER_ITER_S = NUM_ENVS * STEPS_PER_ITER * DT   # ≈ 1966 s d'expérience-robot / itération


def _font(size, bold=False):
    for p in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def human_duration(seconds):
    """Secondes -> texte lisible en jours / mois / années."""
    days = seconds / 86400.0
    if days < 1:
        return f"{seconds/3600.0:.1f} heures"
    if days < 60:
        return f"{days:.1f} jours"
    months = days / 30.44
    if months < 24:
        return f"{months:.1f} mois"
    return f"{days/365.25:.2f} ans"


def load_scalars(log_dir, cap=None):
    """Récupère les séries TensorBoard (itération -> valeur), plafonnées à `cap`.

    FUSION de TOUS les fichiers events : une reprise (`resume_from`) après interruption
    ouvre un NOUVEAU fichier events ; ne lire que le premier couperait la courbe à
    l'interruption (or on veut une courbe CONTINUE — la reprise doit se voir comme un
    entraînement d'un seul tenant). On concatène tout, on déduplique par itération (la
    valeur la plus RÉCENTE gagne), puis on trie. Le filtre `cap` (≤ pic) retire ce qui
    dépasse le segment, donc la fusion reste propre même si un 2ᵉ run déborde.
    Renvoie aussi les wall_time (pour estimer le temps GPU réel par itération)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    evs = sorted(glob.glob(f"{log_dir}/events.out.*"))
    accs = []
    for e in (evs or [log_dir]):
        a = EventAccumulator(e, size_guidance={"scalars": 0})
        a.Reload()
        accs.append(a)
    out = {}
    for tag in ("Train/mean_reward", "Train/mean_episode_length", "Episode/n_success"):
        merged = {}                                   # itération -> (valeur, wall_time)
        for a in accs:
            if tag in a.Tags().get("scalars", []):
                for e in a.Scalars(tag):
                    merged[e.step] = (e.value, e.wall_time)   # le dernier écrit gagne
        if not merged:
            continue
        step = np.array(sorted(merged))
        val = np.array([merged[s][0] for s in step])
        wall = np.array([merged[s][1] for s in step])
        if cap is not None:
            m = step <= cap
            step, val, wall = step[m], val[m], wall[m]
        out[tag] = (step, val)
        out[tag + "/wall"] = wall
    return out


# métriques de courbe disponibles : (tag TensorBoard, titre, format valeur)
CURVE_METRICS = {
    "reward": ("Train/mean_reward",
               "Récompense moyenne par épisode  (↑ = il apprend)", "{:.2f}"),
    "episode_length": ("Train/mean_episode_length",
                       "Durée avant chute (pas)  (↑ = il reste debout)", "{:.0f}"),
    "success": ("Episode/n_success",
                "Taux de réussite : but atteint sans collision  (↑)", "{:.0%}"),
}


def curve_panel(scalars, cur_iter, max_iter, metric="reward", offset=0, note=None):
    """Panneau courbe (RGB HxWx3) : métrique choisie, remplie jusqu'à cur_iter.

    `offset` décale l'axe X en espace AFFICHÉ (0-based) : un segment warm-starté
    (course/terrain) a des itérations absolues 2500+, on montre 0..N pour la lisibilité.
    `metric` : 'reward' (marche) ou 'episode_length' (course : le reward décline près
    des limites moteur, la durée-avant-chute raconte mieux « il court sans tomber »)."""
    tag, title, vfmt = CURVE_METRICS[metric]
    steps, val = scalars[tag]
    steps = steps - offset
    ci = cur_iter - offset
    fig = plt.figure(figsize=(CURVE_W / 100, ROBOT_H / 100), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0e1117")

    # tracé complet en fantôme + portion apprise en vert vif
    ax.plot(steps, val, color="#2a3346", lw=1.5)
    m = steps <= ci
    ax.plot(steps[m], val[m], color="#22e06a", lw=2.6)
    if m.any():
        ax.scatter([steps[m][-1]], [val[m][-1]], s=70, color="#22e06a",
                   zorder=5, edgecolors="white", linewidths=1.2)
        ax.annotate(vfmt.format(val[m][-1]), (steps[m][-1], val[m][-1]),
                    textcoords="offset points", xytext=(-8, 10),
                    color="#22e06a", fontsize=13, ha="right", weight="bold")

    # annotation optionnelle pointant le BOUT de la courbe (ex. LiDAR : « ≈40 % non optimal »)
    if note:
        ax.annotate(note, xy=(steps[-1], val[-1]),
                    xytext=(0.40, 0.34), textcoords="axes fraction",
                    color="#ffb454", fontsize=11, weight="bold", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="#ffb454", lw=1.5,
                                    connectionstyle="arc3,rad=-0.2"))

    ax.set_xlim(0, (max_iter - offset) * 1.05)   # marge droite : marqueur final non rogné
    lo, hi = float(np.min(val)), float(np.max(val))
    pad = 0.1 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title, color="#e6edf3", fontsize=13, pad=10)
    ax.set_xlabel("itération d'entraînement", color="#9aa4b2", fontsize=11)
    ax.tick_params(colors="#5a6675", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#2a3346")
    ax.grid(True, color="#161b24", lw=0.8)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def compose(robot_rgb, curve_rgb, cur_iter, max_iter, gpu_seconds, label="MARCHER",
            sublabel=None, offset=0):
    """Assemble robot | courbe + bandeau. cur_iter/max_iter ABSOLUS ; `offset` les ramène
    en espace affiché (segment warm-starté)."""
    disp, max_disp = cur_iter - offset, max_iter - offset
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "#0e1117")
    canvas.paste(Image.fromarray(robot_rgb).resize((ROBOT_W, ROBOT_H)), (0, 0))
    canvas.paste(Image.fromarray(curve_rgb).resize((CURVE_W, ROBOT_H)), (ROBOT_W, 0))

    d = ImageDraw.Draw(canvas)
    exp_s = disp * EXP_PER_ITER_S
    f_big = _font(30, bold=True)
    f_sm = _font(19)
    y0 = ROBOT_H + 18
    # label COURT (sinon il chevauche la colonne temps à droite, x=360)
    d.text((28, y0), label, font=f_big, fill="#22e06a")
    sub = sublabel if sublabel is not None else f"itération {disp} / {max_disp}"
    d.text((28, y0 + 44), sub, font=f_sm, fill="#9aa4b2")
    # colonne droite : temps GPU vs expérience-robot
    line1 = f"RX 9070 — calcul écoulé : {human_duration(gpu_seconds)}"
    line2 = f"≈ {human_duration(exp_s)} d'expérience-robot (4096 en parallèle)"
    d.text((360, y0), line1, font=f_sm, fill="#e6edf3")
    d.text((360, y0 + 40), line2, font=f_big, fill="#ffd34d")
    # légende réalisme, petit texte gris en bas du bandeau
    f_cap = _font(14)
    d.text((28, CANVAS_H - 26), REALISM_CAPTION, font=f_cap, fill="#6b7686")
    return np.asarray(canvas)


# fractions FRONT-LOADED de la plage affichée : denses au début (là où l'apprentissage
# se joue), espacées ensuite. Indépendant de la granularité de SAUVEGARDE (save_interval
# fin à 25 pour capturer le pic ≠ nombre de clips montrés) et de la LONGUEUR du run
# (un run court à pic précoce, ex. escalier, garde le même profil « dense au début »).
CKPT_FRACTIONS = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.28, 0.40, 0.56, 0.76, 1.0]


def pick_checkpoints(log_dir, cap=None, offset=0, min_disp=0):
    """Checkpoints à rejouer, DENSES au début puis espacés (profil constant quel que soit
    save_interval / la longueur). `offset` ramène les itérations absolues d'un segment
    warm-starté (2500+) en espace affiché 0-based pour l'échantillonnage ; on renvoie les
    numéros ABSOLUS (pour charger model_N). `cap` = pic → dernier clip = meilleur état.
    `min_disp` (affiché) > 0 : on garde UN clip d'ouverture (état « pas encore appris »)
    puis on concentre l'échantillonnage sur [min_disp, max] — évite les longues plages
    « rien ne se passe » au début d'un skill warm-starté (ex. assise : reste debout un
    moment avant de s'asseoir → sinon le spectateur s'ennuie)."""
    its = sorted(int(m.group(1)) for f in glob.glob(f"{log_dir}/model_*.pt")
                 if (m := re.search(r"model_(\d+)\.pt$", f)))
    if cap is not None:
        its = [i for i in its if i <= cap]
    if not its:
        return []
    max_disp = its[-1] - offset
    avail = {i - offset: i for i in its}          # itération affichée -> absolue
    keep = []
    first = min(avail)
    if min_disp > 0:
        keep.append(avail[first])                 # clip d'ouverture (contraste "avant")
    lo = max(min_disp, first)
    for fr in CKPT_FRACTIONS:
        d = min(avail, key=lambda x: abs(x - (lo + fr * (max_disp - lo))))  # palier dispo le + proche
        if avail[d] not in keep:
            keep.append(avail[d])
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--exp_name", default="go2-walking")
    ap.add_argument("--curve-only", action="store_true", help="test courbe sans Genesis")
    ap.add_argument("--vx", type=float, default=1.0, help="vitesse avant commandée au rendu")
    ap.add_argument("--label", default="MARCHER", help="libellé du segment (ex. COURIR)")
    ap.add_argument("--iter-offset", type=int, default=0,
                    help="itération absolue du 1er checkpoint du segment (warm-start). "
                         "Ex. course warm-startée à 2499 → --iter-offset 2500 pour un axe 0-based.")
    ap.add_argument("--curve-metric", default="reward", choices=list(CURVE_METRICS),
                    help="'reward' (marche) ou 'episode_length' (course : le reward décline "
                         "près des limites moteur, la durée-avant-chute raconte mieux la course).")
    ap.add_argument("--secs", type=float, default=SECS_PER_CKPT,
                    help="durée du clip par checkpoint (s) — allonger pour les postures "
                         "phasées (coucher/assise ~3 s + immobilisation)")
    ap.add_argument("--cap", type=int, default=2499,
                    help="itération max du segment (ignore les checkpoints/events au-delà, "
                         "ex. robustification). Passer -1 pour tout inclure.")
    ap.add_argument("--min-iter", type=int, default=0, dest="min_iter",
                    help="itération AFFICHÉE en dessous de laquelle on saute les clips "
                         "(garde 1 clip d'ouverture) : concentre le rendu sur la phase "
                         "intéressante d'un skill warm-starté (ex. assise). 0 = tout montrer.")
    args = ap.parse_args()
    log_dir = f"logs/{args.exp_name}"
    cap = None if args.cap < 0 else args.cap
    offset = args.iter_offset
    metric = args.curve_metric
    metric_tag = CURVE_METRICS[metric][0]

    scalars = load_scalars(log_dir, cap=cap)
    if metric_tag not in scalars:
        raise SystemExit(f"Pas encore de métrique {metric_tag} dans les journaux.")
    steps_m, _ = scalars[metric_tag]
    max_iter = int(steps_m[-1])

    # temps GPU réel par itération : mesuré sur les wall_time TensorBoard (robuste,
    # sans dépendre d'un fichier log externe). Fallback 1.2 s/iter (mesure RX 9070).
    gpu_per_iter = 1.2
    wall = scalars.get(metric_tag + "/wall")
    if wall is not None and len(wall) > 1 and steps_m[-1] > steps_m[0]:
        gpu_per_iter = float((wall[-1] - wall[0]) / (steps_m[-1] - steps_m[0]))

    ckpts = pick_checkpoints(log_dir, cap=cap, offset=offset, min_disp=args.min_iter)
    # Le compteur affiché est "disp / max_disp" = (k−offset) / (max_iter−offset). `max_iter`
    # vient du dernier SCALAR loggé, mais un segment finetuné (course : clip ARRÊT au pic 4025)
    # peut avoir un DERNIER CHECKPOINT au-delà du dernier scalar retenu → "1525 / 1084".
    # On garantit dénominateur ≥ numérateur en couvrant le dernier checkpoint réellement rendu.
    if ckpts:
        max_iter = max(max_iter, ckpts[-1])
    print(f"checkpoints retenus : {ckpts}  |  max_iter={max_iter}  |  offset={offset}  "
          f"|  metric={metric}  |  {gpu_per_iter:.2f}s/iter")

    os.makedirs("videos", exist_ok=True)
    import imageio.v2 as imageio
    out = f"videos/segment_{args.exp_name}.mp4"
    writer = imageio.get_writer(out, fps=FPS, macro_block_size=1)

    if args.curve_only:
        # test rapide : image robot factice (dégradé) pour valider courbe + bandeau
        dummy = np.tile(np.linspace(20, 60, ROBOT_H, dtype=np.uint8)[:, None, None], (1, ROBOT_W, 3))
        for k in ckpts:
            cur = compose(dummy, curve_panel(scalars, k, max_iter, metric, offset), k, max_iter,
                          (k - offset) * gpu_per_iter, label=args.label, offset=offset)
            Image.fromarray(cur).save(f"videos/_curvecheck_{args.exp_name}_{k}.png")
            for _ in range(int(FPS * 1.0)):
                writer.append_data(cur)
        writer.close()
        print(f"🎥 (curve-only) {out}")
        return

    # ---- rendu complet avec Genesis ----
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner

    gs.init(backend=gs.cpu, logging_level="warning")
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    # env TERRAIN si le segment a un relief (spawn sur tuile à la bonne altitude) ;
    # sinon env plat standard. La détection sur cfgs.pkl rend le rendu générique.
    # SE RELEVER : env custom (le robot part COUCHÉ, orientation aléatoire) — sans lui
    # le rendu partirait DEBOUT et la policy de lever n'aurait aucun sens. Les autres
    # postures (se coucher/s'asseoir) partent debout comme l'env de base → rien à faire.
    if env_cfg.get("terrain_cfg"):
        from go2_terrain_env import Go2TerrainEnv as Go2Env
    elif "getup" in args.exp_name:
        from go2_getup_env import Go2GetupEnv as Go2Env
    else:
        from go2_env import Go2Env
    reward_cfg["reward_scales"] = {}
    env_cfg["resampling_time_s"] = 1e9        # NE PAS re-tirer de commande : marche tout droit
    # NE PAS réinitialiser quand il tombe : on VEUT voir le robot s'effondrer aux
    # premiers checkpoints (preuve visuelle qu'il n'a pas encore appris). Seul le
    # reset au DÉBUT de chaque clip (env.reset() dans la boucle) le remet debout.
    env_cfg["termination_if_pitch_greater_than"] = 1.0e4
    env_cfg["termination_if_roll_greater_than"] = 1.0e4
    # RENDU CLEAN : on ISOLE le comportement appris → on coupe les perturbations
    # d'entraînement (poussées, friction randomisée, charge, bruit capteurs). Sinon un
    # robot bousculé aléatoirement rendrait la progression illisible. Les LIMITES
    # MOTEUR restent actives (couple/vitesses bornés = comportement honnête).
    env_cfg["push_interval_s"] = 0
    env_cfg["push_vel"] = 0.0
    env_cfg["friction_range"] = None
    env_cfg["payload_rand_kg"] = 0.0
    env_cfg["obs_noise"] = None
    env_cfg["zero_cmd_frac"] = 0.0
    env = Go2Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False,
                 camera_cfg=[{"res": (ROBOT_W, ROBOT_H), "fov": 40}])
    env.max_episode_length = int(1e9)
    cam = env.cams[0]
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    import torch
    n_frames = int(FPS * args.secs)

    is_terrain = bool(env_cfg.get("terrain_cfg"))

    def place_on_terrain_center():
        # SPAWN DÉTERMINISTE au CENTRE d'une tuile (correctif salto escalier 22/07).
        # Au rendu (1 env), le _reset_idx terrain tire une tuile ALÉATOIRE avec offset →
        # parfois près d'un bord, et 3 s de montée franchissent la JONCTION entre deux
        # tuiles (chaque escalier monte indépendamment → falaise de ~1 m au raccord) : le
        # robot marche dans le vide → salto. La policy est AVEUGLE (proprioception seule),
        # elle ne peut pas éviter un bord invisible → le fix est ici, pas dans un finetune.
        # On le pose reculé sur la tuile (x=3 m) face à +x : ~4-5 m de montée devant lui,
        # arrêt final encore à >1,5 m du bord. y centré = marge latérale maximale.
        sub = env.sub_m
        cx = torch.tensor([3.0], device=env.device)
        cy = torch.tensor([sub * 0.5], device=env.device)
        xy = torch.stack([cx, cy], dim=1)
        z = env._terrain_h(xy) + env_cfg["base_init_pos"][2]
        q = torch.zeros((1, 7 + env.num_actions), dtype=gs.tc_float, device=env.device)
        q[0, 0], q[0, 1], q[0, 2] = cx[0], cy[0], z[0]
        q[0, 3] = 1.0                                   # quat neutre : face à +x (monte)
        q[0, 7:] = env.init_dof_pos
        env.robot.set_qpos(q, envs_idx=torch.tensor([0], device=env.device),
                           zero_velocity=True, skip_forward=True)
        env.base_pos[0] = q[0, :3]
        env._update_observation()
        return env.get_observations()

    def force_cmd():
        env.commands[0, 0] = args.vx
        env.commands[0, 1] = 0.0
        env.commands[0, 2] = 0.0

    def render_frames(n, obs, panel, k, gpu_s, cmd_vx, label, sublabel=None):
        for _ in range(n):
            env.commands[0, 0] = cmd_vx
            env.commands[0, 1] = 0.0
            env.commands[0, 2] = 0.0
            act = policy(obs)
            obs = env.step(act)[0]
            env.commands[0, 0] = cmd_vx
            env.commands[0, 1] = 0.0
            env.commands[0, 2] = 0.0
            x = float(env.base_pos[0, 0]); y = float(env.base_pos[0, 1]); z = float(env.base_pos[0, 2])
            # caméra qui suit AUSSI en Z (indispensable sur terrain : marches/pentes
            # élèvent le robot ; sur sol plat z≈0,4 → cadrage identique à avant).
            cam.set_pose(pos=(x - 1.4, y - 3.0, z + 1.1), lookat=(x + 0.6, y, z - 0.1), up=(0, 0, 1))
            frame = cam.render()[0]
            writer.append_data(compose(frame, panel, k, max_iter, gpu_s, label=label,
                                       sublabel=sublabel, offset=offset))
        return obs

    for k in ckpts:
        runner.load(f"{log_dir}/model_{k}.pt")
        policy = runner.get_inference_policy(device=gs.device)
        panel = curve_panel(scalars, k, max_iter, metric, offset)   # constant pour ce checkpoint
        gpu_s = (k - offset) * gpu_per_iter
        obs = env.reset()
        if is_terrain:
            obs = place_on_terrain_center()   # spawn centré déterministe (anti-chute-de-bord)
        with torch.no_grad():
            obs = render_frames(n_frames, obs, panel, k, gpu_s, args.vx, args.label)
            # sur le DERNIER checkpoint d'une LOCOMOTION (vx≠0) : couper la commande ->
            # le robot s'arrête, pattes figées, posture droite (arrêt maîtrisé demandé).
            # Pour les POSTURES (vx=0 : se coucher/asseoir/relever), la commande est déjà
            # nulle tout du long → un clip « ARRÊT posture droite » serait redondant ET
            # faux (le robot est couché/assis, pas droit) : on ne l'ajoute pas.
            if k == ckpts[-1] and abs(args.vx) > 1e-6:
                render_frames(int(FPS * 3.0), obs, panel, k, gpu_s, 0.0,
                              "ARRÊT", sublabel="pattes figées, posture droite")
        print(f"  checkpoint {k}: rendu ok")
    writer.close()
    print(f"🎥 {out}")


if __name__ == "__main__":
    main()
