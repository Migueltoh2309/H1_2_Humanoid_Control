source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd /home/mito/Github/H1_2_Humanoid_Control/h1_2_algoritms/results/bc/scripts_diagnostico
SEEDS=$(seq -s, 5000 5023) VARS='{"fifo":{"pol":"fifo"},"exp_def":{},"sep_t1":{"sep_t":1.0},"sep_t1_y15":{"sep_t":1.0,"sep_y":0.15},"ventanas":{"windows":{"right":[-0.34,0.08],"left":[-0.08,0.34]}}}' python experto_var.py 2>&1 | grep -E " dep |Error|Trace"
