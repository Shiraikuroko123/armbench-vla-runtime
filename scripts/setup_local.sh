#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="python3"
with_vla=0
skip_models=0
menagerie_commit="71f066ad0be9cd271f7ed58c030243ef157af9f4"

while (($#)); do
  case "$1" in
    --with-vla) with_vla=1 ;;
    --skip-models) skip_models=1 ;;
    --python)
      shift
      python_command="${1:?--python requires an executable}"
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

venv_root="$project_root/.venv"
venv_python="$venv_root/bin/python"
menagerie_root="$project_root/.cache/mujoco_menagerie"
panda_scene="$menagerie_root/franka_emika_panda/scene.xml"

if [[ ! -x "$venv_python" ]]; then
  "$python_command" -m venv "$venv_root"
fi
"$venv_python" -m pip install --upgrade pip

install_target=".[test]"
if ((with_vla)); then
  install_target=".[test,vla]"
fi
(
  cd "$project_root"
  "$venv_python" -m pip install -e "$install_target"
)

if ((!skip_models)) && [[ ! -f "$panda_scene" ]]; then
  if [[ -e "$menagerie_root" ]]; then
    echo "Incomplete model cache at $menagerie_root. Move it aside and rerun setup." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$menagerie_root")"
  git clone --filter=blob:none --no-checkout \
    https://github.com/google-deepmind/mujoco_menagerie.git "$menagerie_root"
  git -C "$menagerie_root" sparse-checkout init --cone
  git -C "$menagerie_root" sparse-checkout set franka_emika_panda
  git -C "$menagerie_root" checkout --detach "$menagerie_commit"
fi

doctor_args=()
if ((with_vla)); then
  doctor_args+=(--require-vla)
fi
"$venv_python" -m armbench doctor "${doctor_args[@]}"

echo
echo "ArmBench is ready."
echo "Python: $venv_python"
echo "Viewer: $venv_python -m armbench mujoco-view --scenario narrow_gate"
