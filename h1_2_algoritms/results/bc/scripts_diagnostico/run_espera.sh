source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd /home/mito/Github/H1_2_Humanoid_Control/h1_2_algoritms/results/bc/scripts_diagnostico
C='"tactil":true,"deposit_pose":[[0.20,-0.40,-0.05],-60.0,0.0],"espera":[0.34,-0.10,0.12]'
T='"timing":{"show_verify":0.05,"deposit":1.05,"home":0.5}'
SEEDS=$(seq -s, 3000 3015) POL=experto PERFILES='{
"e_t07":{'"$C"',"reach_tau_scale":0.7,"grasp_kwargs":{"close_max_vrel":0.05},'"$T"'},
"e_t05":{'"$C"',"reach_tau_scale":0.5,"grasp_kwargs":{"close_max_vrel":0.05},'"$T"'},
"e_t05_ct":{'"$C"',"reach_tau_scale":0.5,"grasp_kwargs":{"close_max_vrel":0.05,"close_touch":true,"lift_height":0.06,"lift_time":0.5},"timing":{"show_verify":0.05,"deposit":1.05,"home":0.5,"close_lift":1.2}}}' python perfiles.py 2>&1 | grep -E "depositadas|Error|Trace"
