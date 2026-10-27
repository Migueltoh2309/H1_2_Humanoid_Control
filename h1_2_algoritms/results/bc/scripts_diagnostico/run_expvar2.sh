source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd /home/mito/Github/H1_2_Humanoid_Control/h1_2_algoritms/results/bc/scripts_diagnostico
SEEDS=$(seq -s, 5024 5047) VARS='{"exp_def":{},"sep_t1_y15":{"sep_t":1.0,"sep_y":0.15}}' python experto_var.py 2>&1 | grep -E " dep |Error|Trace"
SEEDS=$(seq -s, 5000 5023) VARS='{"sep_t1_y10":{"sep_t":1.0,"sep_y":0.10},"sep_t05_y15":{"sep_t":0.5,"sep_y":0.15}}' python experto_var.py 2>&1 | grep -E " dep |Error|Trace"
