source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd /home/mito/Github/H1_2_Humanoid_Control/h1_2_algoritms/results/bc/scripts_diagnostico
for p in tactil base; do PERF=$p SEEDS=$(seq -s, 3000 3015) python rasgos.py 2>&1 | grep -E "^(tactil|base) |Error"; done
