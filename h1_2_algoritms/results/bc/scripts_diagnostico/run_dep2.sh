source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd /home/mito/Github/H1_2_Humanoid_Control/h1_2_algoritms/results/bc/scripts_diagnostico
DC='"deposit_pose":[[0.20,-0.40,-0.05],-60.0,0.0]'
SEEDS=$(seq -s, 3000 3015) POL=experto PERFILES='{"corto":{"tactil":true,"timing":{"show_verify":0.05,"deposit":1.05,"home":0.5},"grasp_kwargs":{"close_max_vrel":0.05},'"$DC"'},"base_dc":{"timing":{"deposit":1.05,"home":0.5},'"$DC"'}}' python perfiles.py 2>&1 | grep -E "depositadas|Error|Trace"
