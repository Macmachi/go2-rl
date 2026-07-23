"""Imprime l'itération du MEILLEUR checkpoint sauvé d'un run (pour --cap au rendu).
Fusionne TOUS les fichiers d'events (un run repris crée un nouveau fichier — ne lire
que le dernier tronquait l'historique et faisait rater le vrai pic, bug du 22/07).
Choisit, parmi les checkpoints présents sur disque, celui qui MAXIMISE la métrique.

    python _peakcap.py <exp> <metric_substr> [min_iter]
    # ex: _peakcap.py go2-terrain-escalier episode_length
    # ex: _peakcap.py go2-run episode_length 3800   (pic APRÈS le finetune d'arrêt)
"""
import glob
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

exp, metric_sub = sys.argv[1], sys.argv[2]
min_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 0

ckpts = sorted(int(m.group(1)) for f in glob.glob(f"logs/{exp}/model_*.pt")
               if (m := re.search(r"model_(\d+)\.pt$", f)))
if not ckpts:
    print(""); sys.exit(0)

d = {}
for f in sorted(glob.glob(f"logs/{exp}/events*")):
    ea = EventAccumulator(f); ea.Reload()
    tags = ea.Tags()["scalars"]
    key = next((t for t in tags if metric_sub in t), None)
    if key is None:
        continue
    for s in ea.Scalars(key):
        d[int(s.step)] = s.value          # les fichiers plus récents écrasent (reprises)
if not d:
    print(ckpts[-1]); sys.exit(0)

def val(c):
    return d.get(c, d[min(d, key=lambda x: abs(x - c))])

# on IGNORE le premier checkpoint (warm-start brut) sauf s'il est seul ; filtre min_iter
cand = [c for c in (ckpts[1:] if len(ckpts) > 1 else ckpts) if c >= min_iter]
if not cand:
    cand = ckpts[-1:]
print(max(cand, key=val))
