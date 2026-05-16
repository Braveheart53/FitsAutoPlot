#!/usr/bin/env bash
# ============================================================================
# setup_py3p8_nrao.sh -- One-shot installer for the FITS/Franks AutoPlot
#                        Python 3.8 environment on an NRAO RHEL 8 host.
#
# What it does (in order):
#   1.  Sanity-check the host (RHEL 8, glibc >= 2.28, x86_64, no sudo needed).
#   2.  Download the latest Miniforge3 installer (which bundles `mamba`)
#       from the official conda-forge GitHub release into $HOME, with a
#       curl -> wget fallback for proxy-restricted NRAO subnets.
#   3.  Install Miniforge3 into $HOME/miniforge3 in batch mode (idempotent:
#       re-runs only update the channels and the env).
#   4.  Create or update a conda env named `Py3p8` pinning Python 3.8 from
#       conda-forge with every runtime package these AutoPlot modules need:
#         numpy, astropy, psutil, qtpy, PySide6, veusz (>=3.4, ships 3.6+)
#         optional: cupy (only when an NVIDIA GPU is detected via nvidia-smi).
#   5.  Write two startup helpers into $HOME/bin:
#         - Py3p8-activate   : sources the env for interactive shells.
#         - Py3p8-autoplot   : launches either FITS_AutoPlot or
#                              Franks_AutoPlot from the cloned repo.
#       Plus a freedesktop .desktop launcher for the NRAO GNOME session.
#   6.  Print a short verification banner showing python / numpy / veusz
#       versions so the user can confirm the install in one glance.
#
# Usage:
#     ./setup_py3p8_nrao.sh             # standard install, autodetect GPU
#     ./setup_py3p8_nrao.sh --no-gpu    # skip the CuPy probe
#     ./setup_py3p8_nrao.sh --update    # just refresh packages in Py3p8
#     ./setup_py3p8_nrao.sh --prefix DIR
#                                       # use a non-default install prefix
#     ./setup_py3p8_nrao.sh --repo DIR  # use a non-default repo location
#                                       # (default: $HOME/GitHub/FitsAutoPlot)
#
# Author : W Wallace (NRAO)            Last updated: 2026-05-16
# Target : RHEL 8.x  /  Python 3.8     Env name   : Py3p8
# ============================================================================
set -Eeuo pipefail

# ---------------------------------------------------------------- arg parsing
GPU_PROBE=auto
DO_UPDATE_ONLY=0
PREFIX="${HOME}/miniforge3"
REPO_DIR="${HOME}/GitHub/FitsAutoPlot"
ENV_NAME="Py3p8"
PY_VERSION="3.8"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-gpu)   GPU_PROBE=no ; shift ;;
        --gpu)      GPU_PROBE=yes ; shift ;;
        --update)   DO_UPDATE_ONLY=1 ; shift ;;
        --prefix)   PREFIX="$2" ; shift 2 ;;
        --repo)     REPO_DIR="$2" ; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0" ; exit 0 ;;
        *)  echo "Unknown argument: $1" >&2 ; exit 2 ;;
    esac
done

# ----------------------------------------------------------- logging helpers
c_reset='\033[0m'; c_blue='\033[1;34m'; c_green='\033[1;32m'
c_yellow='\033[1;33m'; c_red='\033[1;31m'
info()  { printf "${c_blue}[info]${c_reset}  %s\n" "$*"; }
ok()    { printf "${c_green}[ ok ]${c_reset}  %s\n" "$*"; }
warn()  { printf "${c_yellow}[warn]${c_reset}  %s\n" "$*"; }
die()   { printf "${c_red}[FAIL]${c_reset}  %s\n" "$*" >&2 ; exit 1; }

# Make sure unexpected errors are visible.
trap 'die "Aborted on line $LINENO (exit $?)."' ERR

# =====================================================================
# 1.  Host sanity checks
# =====================================================================
info "Checking host compatibility ..."

if [[ "$(uname -s)" != "Linux" ]]; then
    die "This installer only targets Linux (uname -s = $(uname -s))."
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    die "Unsupported arch: $(uname -m).  Only x86_64 is supported."
fi
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    info "OS: ${NAME:-?} ${VERSION_ID:-?}"
    case "${ID:-}${ID_LIKE:-}" in
        *rhel*|*fedora*|*centos*|*rocky*|*almalinux*) : ;;
        *) warn "Host is not in the RHEL family -- proceeding anyway." ;;
    esac
else
    warn "/etc/os-release not readable; skipping OS check."
fi

# glibc check (Miniforge needs >= 2.17; RHEL 8 ships 2.28).
if command -v ldd >/dev/null 2>&1; then
    glibc_ver=$(ldd --version 2>/dev/null | head -n1 | awk '{print $NF}') || glibc_ver="?"
    info "glibc: ${glibc_ver}"
fi

# Free-space check: Miniforge + Py3p8 + Qt/Veusz needs ~3 GB.
free_kb=$(df -k --output=avail "$HOME" | tail -n1 | tr -d ' ')
if (( free_kb < 4*1024*1024 )); then
    warn "Less than 4 GB free in \$HOME (~$((free_kb/1024/1024)) GB).  Install may fail."
else
    ok "Free space in \$HOME looks sufficient (~$((free_kb/1024/1024)) GB)."
fi

# =====================================================================
# 2.  Pick a downloader (curl -> wget)
# =====================================================================
fetch() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --silent --show-error --retry 3 --retry-delay 5 \
             -o "$out" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --quiet --tries=3 --timeout=30 -O "$out" "$url"
    else
        die "Neither curl nor wget is available."
    fi
}

# =====================================================================
# 3.  Download + install Miniforge3 (bundles mamba)
# =====================================================================
INSTALLER="${HOME}/Miniforge3-Linux-x86_64.sh"
INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

if [[ -x "${PREFIX}/bin/mamba" ]]; then
    ok "Miniforge already installed at ${PREFIX}."
else
    info "Downloading Miniforge3 (bundles mamba) ..."
    info "  URL: ${INSTALLER_URL}"
    fetch "${INSTALLER_URL}" "${INSTALLER}"
    ok   "Downloaded $(du -h "${INSTALLER}" | awk '{print $1}') to ${INSTALLER}"

    info "Installing into ${PREFIX} (batch mode, no PATH edits yet) ..."
    bash "${INSTALLER}" -b -p "${PREFIX}"
    ok   "Miniforge installed."
    rm -f "${INSTALLER}"
fi

# Source the conda shell hook for the rest of this script.
# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"
# Make `mamba` available in the current shell.
if [[ -f "${PREFIX}/etc/profile.d/mamba.sh" ]]; then
    # shellcheck disable=SC1091
    source "${PREFIX}/etc/profile.d/mamba.sh"
fi

# =====================================================================
# 4.  Configure channels (conda-forge first, strict priority)
# =====================================================================
info "Configuring channels (conda-forge first, strict priority) ..."
"${PREFIX}/bin/conda" config --system --set channel_priority strict || true
"${PREFIX}/bin/conda" config --add channels conda-forge --file "${HOME}/.condarc" 2>/dev/null || \
    "${PREFIX}/bin/conda" config --add channels conda-forge
"${PREFIX}/bin/conda" config --set channel_priority strict
ok "Channel priority set."

# =====================================================================
# 5.  GPU probe (optional CuPy install)
# =====================================================================
GPU_PKGS=()
if [[ "${GPU_PROBE}" == "no" ]]; then
    info "GPU probe skipped (--no-gpu)."
elif [[ "${GPU_PROBE}" == "yes" ]]; then
    GPU_PKGS+=("cupy")
    info "GPU forced on (--gpu); cupy will be installed."
else
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        ok "NVIDIA GPU detected -- cupy will be added to the env."
        GPU_PKGS+=("cupy")
    else
        info "No NVIDIA GPU detected; cupy will be skipped (modules already guard the import)."
    fi
fi

# =====================================================================
# 6.  Create or update the Py3p8 environment
# =====================================================================
# Runtime packages required by the AutoPlot modules:
#   - python=3.8         (conda-forge still ships maintenance builds)
#   - numpy              numeric arrays
#   - astropy            FITS I/O + units
#   - psutil             memory-aware cache / monitor
#   - qtpy               Qt abstraction layer used by the modules
#   - pyside6            concrete Qt6 backend behind qtpy
#   - veusz              the plotting application + Python API
#                        (conda-forge ships >= 3.6; works for 3.4 AND 4.1
#                         compatibility paths in the modules)
#   - pip, setuptools    runtime safety net for anything missing
CORE_PKGS=(
    "python=${PY_VERSION}"
    "numpy"
    "astropy"
    "psutil"
    "qtpy"
    "pyside6"
    "veusz"
    "pip"
    "setuptools"
    "wheel"
)

if "${PREFIX}/bin/conda" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ "${DO_UPDATE_ONLY}" -eq 1 ]]; then
        info "Updating existing env '${ENV_NAME}' ..."
    else
        info "Env '${ENV_NAME}' already exists; refreshing packages ..."
    fi
    "${PREFIX}/bin/mamba" install -n "${ENV_NAME}" -y -c conda-forge \
        "${CORE_PKGS[@]}" "${GPU_PKGS[@]}"
else
    info "Creating env '${ENV_NAME}' with Python ${PY_VERSION} ..."
    "${PREFIX}/bin/mamba" create -n "${ENV_NAME}" -y -c conda-forge \
        "${CORE_PKGS[@]}" "${GPU_PKGS[@]}"
fi
ok "Env '${ENV_NAME}' ready."

# =====================================================================
# 7.  Verify versions
# =====================================================================
info "Verifying installed package versions ..."
"${PREFIX}/bin/conda" run -n "${ENV_NAME}" python - <<'PY'
import sys, importlib
mods = ["numpy", "astropy", "psutil", "qtpy", "PySide6", "veusz"]
print("python :", sys.version.split()[0])
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"{m:<8}: {v}")
    except Exception as exc:
        print(f"{m:<8}: IMPORT FAILED -- {exc}")
try:
    import cupy
    print(f"cupy    : {cupy.__version__}  (GPU optional)")
except Exception:
    print("cupy    : not installed  (CPU-only fallback active)")
PY
ok "Verification complete."

# =====================================================================
# 8.  Write startup helpers into $HOME/bin
# =====================================================================
mkdir -p "${HOME}/bin"

ACTIVATE_SCRIPT="${HOME}/bin/Py3p8-activate"
cat > "${ACTIVATE_SCRIPT}" <<EOF
#!/usr/bin/env bash
# Source this file to drop into the Py3p8 environment.
# Usage:  source ~/bin/Py3p8-activate
# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
EOF
chmod +x "${ACTIVATE_SCRIPT}"
ok "Wrote ${ACTIVATE_SCRIPT}"

LAUNCH_SCRIPT="${HOME}/bin/Py3p8-autoplot"
cat > "${LAUNCH_SCRIPT}" <<EOF
#!/usr/bin/env bash
# Launch FITS_AutoPlot or Franks_AutoPlot inside the Py3p8 env.
# Usage:  Py3p8-autoplot fits     # FITS_AutoPlot GUI
#         Py3p8-autoplot franks   # Franks_AutoPlot GUI
#         Py3p8-autoplot shell    # interactive Python shell in the env
set -Eeuo pipefail
target="\${1:-fits}"
# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${REPO_DIR}" || {
    echo "Repo not found at ${REPO_DIR}." >&2
    echo "Clone or symlink it there, or re-run setup with --repo DIR." >&2
    exit 1
}
case "\${target}" in
    fits|FITS)     exec python FITS_AutoPlot.py "\${@:2}" ;;
    franks|Franks) exec python Franks_AutoPlot.py "\${@:2}" ;;
    shell|repl)    exec python ;;
    *) echo "Unknown target: \${target} (try fits|franks|shell)" >&2 ; exit 2 ;;
esac
EOF
chmod +x "${LAUNCH_SCRIPT}"
ok "Wrote ${LAUNCH_SCRIPT}"

# Freedesktop launcher for the GNOME app grid (NRAO standard desktop).
DESKTOP_DIR="${HOME}/.local/share/applications"
mkdir -p "${DESKTOP_DIR}"
cat > "${DESKTOP_DIR}/Py3p8-FITS_AutoPlot.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=FITS AutoPlot (Py3p8)
Comment=NRAO 1PPS-delta FITS plotter
Exec=${LAUNCH_SCRIPT} fits
Terminal=false
Categories=Science;DataVisualization;
EOF
cat > "${DESKTOP_DIR}/Py3p8-Franks_AutoPlot.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Franks AutoPlot (Py3p8)
Comment=Franks-format time-series plotter
Exec=${LAUNCH_SCRIPT} franks
Terminal=false
Categories=Science;DataVisualization;
EOF
ok "Wrote freedesktop launchers in ${DESKTOP_DIR}"

# =====================================================================
# 9.  PATH hint for interactive shells
# =====================================================================
if ! grep -qs "Py3p8-activate" "${HOME}/.bashrc" 2>/dev/null; then
    cat >> "${HOME}/.bashrc" <<EOF

# >>> Py3p8 / FITS AutoPlot environment (added by setup_py3p8_nrao.sh) >>>
# Run \`source ~/bin/Py3p8-activate\` to enter the env, or call
# \`Py3p8-autoplot fits\` / \`Py3p8-autoplot franks\` to launch a GUI.
case ":\$PATH:" in
    *":\$HOME/bin:"*) : ;;
    *) export PATH="\$HOME/bin:\$PATH" ;;
esac
# <<< Py3p8 / FITS AutoPlot environment <<<
EOF
    ok "Appended PATH hint to ~/.bashrc"
else
    info "~/.bashrc already references Py3p8-activate; leaving it alone."
fi

# =====================================================================
# 10.  Final banner
# =====================================================================
echo
printf "${c_green}=============================================================${c_reset}\n"
printf "${c_green}  Py3p8 environment ready.${c_reset}\n"
printf "${c_green}=============================================================${c_reset}\n"
cat <<EOF

  Install prefix    : ${PREFIX}
  Env name          : ${ENV_NAME}    (python ${PY_VERSION})
  Repo expected at  : ${REPO_DIR}
  Activate (shell)  : source ~/bin/Py3p8-activate
  Launch FITS GUI   : ~/bin/Py3p8-autoplot fits
  Launch Franks GUI : ~/bin/Py3p8-autoplot franks
  Update later      : ${0##*/} --update

  If the repo isn't yet on this host, clone it:
      mkdir -p "$(dirname "${REPO_DIR}")"
      git clone <repo-url> "${REPO_DIR}"

  GUI launchers will appear in the GNOME app grid after the next login
  (or after running:  update-desktop-database ~/.local/share/applications)

EOF
