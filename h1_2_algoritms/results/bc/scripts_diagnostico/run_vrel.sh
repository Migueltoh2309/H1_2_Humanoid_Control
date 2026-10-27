source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd /home/mito/Github/H1_2_Humanoid_Control/h1_2_algoritms/results/bc/scripts_diagnostico
SEEDS=$(seq -s, 3000 3015) POL=experto PERFILES='{"t_v05":{"tactil":true,"timing":{"show_verify":0.1,"deposit":1.95},"grasp_kwargs":{"close_max_vrel":0.05}},"t_v03":{"tactil":true,"timing":{"show_verify":0.1,"deposit":1.95},"grasp_kwargs":{"close_max_vrel":0.03}},"b_v05":{"grasp_kwargs":{"close_max_vrel":0.05}}}' python razones.py 2>&1 | grep -E "dep |Error|Trace"
