source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd /home/mito/Github/H1_2_Humanoid_Control/h1_2_algoritms/results/bc/scripts_diagnostico
C='"tactil":true,"deposit_pose":[[0.20,-0.40,-0.05],-60.0,0.0],"espera":[0.34,-0.10,0.12],"grasp_kwargs":{"close_max_vrel":0.05,"close_touch":true,"lift_height":0.06,"lift_time":0.5}'
SEEDS=$(seq -s, 3000 3015) POL=experto PERFILES='{
"t05_cal":{'"$C"',"reach_tau_scale":0.5,"timing":{"reach_extra":0.37,"close_lift":1.35,"show_verify":0.05,"deposit":1.15,"home":0.4}},
"t04_cal":{'"$C"',"reach_tau_scale":0.4,"timing":{"reach_extra":0.45,"close_lift":1.35,"show_verify":0.05,"deposit":1.15,"home":0.4}}}' python perfiles_v.py 2>&1 | grep -E "depositadas|vmax|Error|Trace"
