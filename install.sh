#!/usr/bin/env bash
# ============================================================================
#  INSTALLATION COMPLÈTE de l'environnement de simulation (GPU AMD / ROCm)
#  — pensée pour débutants : une seule commande, tout est isolé dans un
#  conteneur distrobox, rien n'est modifié sur ton système.
#
#      ./install.sh
#
#  Ce que fait le script :
#    1. vérifie que distrobox + podman (ou docker) sont installés ;
#    2. crée le conteneur "genesis-box" depuis l'image officielle AMD
#       rocm/pytorch (PyTorch ROCm déjà compilé dedans — le plus dur est fait) ;
#    3. crée un environnement virtuel Python (~/venvs/genesis) qui HÉRITE du
#       PyTorch ROCm de l'image (pas de téléchargement de wheels exotiques) ;
#    4. installe Genesis + les dépendances du projet (versions épinglées) ;
#    5. vérifie que le GPU est bien visible.
#
#  Personnalisable par variables d'environnement :
#      BOX_NAME=mon-box  BOX_IMAGE=...  VENV_DIR=...  ./install.sh
#
#  Désinstallation totale : ./uninstall.sh
# ============================================================================
set -e

BOX="${BOX_NAME:-genesis-box}"
IMAGE="${BOX_IMAGE:-docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
VENV="${VENV_DIR:-$HOME/venvs/genesis}"

echo "== [1/5] Vérification des prérequis =="
if ! command -v distrobox >/dev/null 2>&1; then
    echo "ERREUR : distrobox n'est pas installé."
    echo "  Fedora :        sudo dnf install distrobox podman"
    echo "  Debian/Ubuntu : sudo apt install distrobox podman"
    echo "  Arch :          sudo pacman -S distrobox podman"
    exit 1
fi
if ! command -v podman >/dev/null 2>&1 && ! command -v docker >/dev/null 2>&1; then
    echo "ERREUR : ni podman ni docker n'est installé (voir ci-dessus)."
    exit 1
fi

echo "== [2/5] Création du conteneur '$BOX' (image ~15 Go, long au 1er téléchargement) =="
if distrobox list 2>/dev/null | grep -q "$BOX"; then
    echo "   conteneur '$BOX' déjà présent — on le réutilise."
else
    # --device /dev/kfd + /dev/dri : accès GPU AMD (ROCm) depuis le conteneur ;
    # keep-groups : conserve tes groupes video/render à l'intérieur.
    distrobox create --name "$BOX" --image "$IMAGE" --yes \
        --additional-flags "--device /dev/kfd --device /dev/dri --group-add keep-groups"
fi

echo "== [3/5] Environnement virtuel Python ($VENV) =="
distrobox enter "$BOX" -- bash -lc "
set -e
if [ ! -f '$VENV/bin/activate' ]; then
    python3 -m venv '$VENV'
    # Héritage du PyTorch préinstallé de l'image via un .pth — PAS
    # --system-site-packages : un venv créé depuis /opt/venv remonterait au
    # python SYSTÈME (sans torch). On DÉTECTE où vit torch : /opt/venv
    # (images ROCm) ou le python système (la plupart des images CUDA).
    TORCH_SP=\$(/opt/venv/bin/python -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null \
             || python3 -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null || true)
    if [ -z \"\$TORCH_SP\" ]; then
        echo 'ERREUR : PyTorch introuvable dans cette image (ni /opt/venv, ni python système).'
        echo '         Utilise une image qui embarque PyTorch (rocm/pytorch, nvcr.io/nvidia/pytorch, …).'
        exit 1
    fi
    VENV_SP=\$('$VENV/bin/python' -c 'import site; print(site.getsitepackages()[0])')
    echo \"\$TORCH_SP\" > \"\$VENV_SP/00_torch_inherit.pth\"
    echo \"   PyTorch hérité de : \$TORCH_SP\"
fi
source '$VENV/bin/activate'
pip install --quiet --upgrade pip
echo '== [4/5] Installation de Genesis + dépendances du projet =='
pip install 'genesis-world==1.2.2' 'rsl-rl-lib==5.4.2' tensorboard imageio imageio-ffmpeg matplotlib
echo '== [5/5] Vérification GPU / versions =='
python - <<'PY'
import torch, genesis, importlib.metadata as md
print('PyTorch  :', torch.__version__)
print('GPU visible (ROCm) :', torch.cuda.is_available())
print('Genesis  :', genesis.__version__)
print('rsl-rl   :', md.version('rsl-rl-lib'))
PY
"

cat <<'EOF'

============================================================
 INSTALLATION TERMINÉE ✔
============================================================
 Pour lancer un entraînement :

   distrobox enter genesis-box
   source ~/venvs/genesis/bin/activate
   cd <dossier-du-projet>
   python go2_train.py -e go2-walking -B 4096 --max_iterations 2500

 Suivre l'apprentissage :   tensorboard --logdir logs
 Rendre la vidéo (APRÈS l'entraînement — jamais pendant !) :
   python pilot_walk_video.py -e go2-walking

 Si "GPU visible : False" : vérifie que ton utilisateur est dans
 les groupes video/render (`sudo usermod -aG video,render $USER`,
 puis déconnecte/reconnecte ta session).
============================================================
EOF
