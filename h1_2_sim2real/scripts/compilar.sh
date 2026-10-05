#!/usr/bin/env bash
# Compila SOLO este workspace (no toca ~/humanoid_ws/build ni install).
set -e
S2R="$(cd "$(dirname "$0")/.." && pwd)"
source "$S2R/entorno.sh"
cd "$S2R"
colcon build --base-paths src "$@"
echo "Listo. En cada terminal:  source $S2R/entorno.sh"
