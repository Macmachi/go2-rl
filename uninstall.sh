#!/usr/bin/env bash
# ============================================================================
#  DÉSINSTALLATION de l'environnement de simulation.
#
#  NE TOUCHE PAS : ton code, logs/ (checkpoints) ni videos/ (rendus).
#
#      ./uninstall.sh              # ce qui n'appartient qu'à ce dépôt
#      ./uninstall.sh --partage    # + le venv commun et le conteneur
#      ./uninstall.sh --image      # + l'image de base (~15 Go)
#      ./uninstall.sh --partage --image -y
#
#  ---------------------------------------------------------------------------
#  POURQUOI LE PARTAGÉ DEMANDE UN DRAPEAU
#
#  Le conteneur et le venv portent des **noms génériques** : d'autres dépôts
#  installés par le même chemin s'en servent aussi. La version précédente les
#  supprimait tous les deux derrière une seule question « Continuer ? », sans
#  jamais dire qu'ils étaient partagés — un simple ménage de fin de projet
#  cassait silencieusement l'environnement des autres.
#
#  Une suppression ne doit jamais énumérer « tout ce que je crois t'appartenir ».
#  Elle énumère **ce qu'on sait remplacer sans coût pour personne d'autre**, et
#  le reste demande un geste explicite.
# ============================================================================
set -e

PARTAGE=0
IMAGE_AUSSI=0
OUI=0
for arg in "$@"; do
    case "$arg" in
        --partage|--shared) PARTAGE=1 ;;
        --image)            IMAGE_AUSSI=1 ;;
        -y|--yes)           OUI=1 ;;
        -h|--help)          sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $arg" >&2; exit 1 ;;
    esac
done

BOX="${BOX_NAME:-genesis-box}"
IMAGE="${BOX_IMAGE:-docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
VENV="${VENV_DIR:-$HOME/venvs/genesis}"
PROJET="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Va être supprimé :"
if [ -d "$PROJET/.venv" ]; then
    echo "  - le venv du dépôt   : $PROJET/.venv"
else
    echo "  - (rien qui n'appartienne à ce seul dépôt)"
fi
if [ "$PARTAGE" -eq 1 ]; then
    echo "  - le venv commun     : $VENV"
    echo "  - le conteneur       : $BOX"
    echo
    echo "  ⚠️  Ces deux-là sont PARTAGÉS. Tout autre dépôt qui s'en sert"
    echo "      cessera de fonctionner jusqu'à sa réinstallation."
fi
[ "$IMAGE_AUSSI" -eq 1 ] && echo "  - l'image de base    : ~15 Go à retélécharger"
echo "  (le code du projet, logs/ et videos/ sont conservés)"

if [ "$PARTAGE" -eq 0 ]; then
    echo
    echo "Le venv commun et le conteneur sont conservés. Pour les enlever aussi :"
    echo "    ./uninstall.sh --partage"
fi

if [ "$OUI" -eq 0 ]; then
    read -r -p $'\nContinuer ? [o/N] ' rep
    [[ "$rep" =~ ^[oOyY] ]] || { echo "Annulé."; exit 0; }
fi

# --- Ce qui n'appartient qu'à ce dépôt --------------------------------------
if [ -d "$PROJET/.venv" ]; then
    rm -rf "$PROJET/.venv"
    echo "venv du dépôt supprimé."
fi

# --- Ce qui est partagé, sur demande explicite seulement ---------------------
if [ "$PARTAGE" -eq 1 ]; then
    if [ -d "$VENV" ]; then
        rm -rf "$VENV"
        echo "venv commun supprimé."
    else
        echo "venv commun déjà absent."
    fi

    if distrobox list 2>/dev/null | grep -q "$BOX"; then
        distrobox rm --force "$BOX"
        echo "conteneur supprimé."
    else
        echo "conteneur déjà absent."
    fi
fi

if [ "$IMAGE_AUSSI" -eq 1 ]; then
    # `rmi` échoue tant qu'un conteneur s'en sert : c'est exactement la garde
    # qu'on veut, et on ne la force jamais.
    (podman rmi "$IMAGE" 2>/dev/null || docker rmi "$IMAGE" 2>/dev/null) \
        && echo "image supprimée." \
        || echo "image introuvable ou encore utilisée — conservée."
fi

echo "Désinstallation terminée."
