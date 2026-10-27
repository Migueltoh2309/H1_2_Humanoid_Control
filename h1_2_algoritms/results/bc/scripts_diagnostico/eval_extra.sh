source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd ~/Github/H1_2_Humanoid_Control/h1_2_algoritms; R=results/bc
for P in corto base; do
  date; echo $P
  python3 demos/bc_faja.py evaluar --escenario exigente --perfil $P --workers 6 --seed 10030 --episodios 60 \
      --out evaluacion_exigente_${P}_s10030.csv > $R/evaluar_exigente_${P}_s10030_log.txt 2>&1 || echo FALLO $P
  tail -5 $R/evaluar_exigente_${P}_s10030_log.txt
done; date
