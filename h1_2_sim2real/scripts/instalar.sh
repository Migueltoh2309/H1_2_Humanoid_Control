#!/usr/bin/env bash
# Una sola vez: SDK de Unitree (Python) y cyclonedds 0.10.2 dentro de terceros/,
# sin instalar nada en ~/venvs/h12 ni en el sistema. Luego compila.
set -e
S2R="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$S2R/terceros"
cd "$S2R/terceros"
[ -d unitree_sdk2_python ] || git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git
source "$HOME/venvs/h12/bin/activate"
# el wheel manylinux de cyclonedds 0.10.2 trae su libddsc: no hace falta CYCLONEDDS_HOME
pip install --no-deps --target "$S2R/terceros/pydeps" cyclonedds==0.10.2 rich-click
"$S2R/scripts/compilar.sh"
