#!/usr/bin/env bash
# ============================================================================
#  DÉSINSTALLATION de l'environnement de simulation.
#  Supprime : le venv Python, le conteneur distrobox et (optionnel) l'image.
#  NE TOUCHE PAS : ton code, logs/ (checkpoints) ni videos/ (rendus).
#
#      ./uninstall.sh
# ============================================================================
set -e

BOX="${BOX_NAME:-genesis-box}"
IMAGE="${BOX_IMAGE:-docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
VENV="${VENV_DIR:-$HOME/venvs/genesis}"

echo "Ce script va supprimer :"
echo "  - le venv            : $VENV"
echo "  - le conteneur       : $BOX"
echo "  (le code du projet, logs/ et videos/ sont conservés)"
read -r -p "Continuer ? [o/N] " rep
[[ "$rep" =~ ^[oOyY]$ ]] || { echo "Annulé."; exit 0; }

if [ -d "$VENV" ]; then
    rm -rf "$VENV"
    echo "venv supprimé."
else
    echo "venv déjà absent."
fi

if distrobox list 2>/dev/null | grep -q "$BOX"; then
    distrobox rm --force "$BOX"
    echo "conteneur supprimé."
else
    echo "conteneur déjà absent."
fi

read -r -p "Supprimer aussi l'image de base (~15 Go, re-téléchargée si tu réinstalles) ? [o/N] " rep
if [[ "$rep" =~ ^[oOyY]$ ]]; then
    (podman rmi "$IMAGE" 2>/dev/null || docker rmi "$IMAGE" 2>/dev/null) && echo "image supprimée." \
        || echo "image introuvable ou utilisée ailleurs."
fi

echo "Désinstallation terminée."
