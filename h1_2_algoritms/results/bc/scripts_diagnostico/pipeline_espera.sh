source /opt/ros/humble/setup.bash && source ~/humanoid_ws/install/setup.bash && source ~/venvs/h12/bin/activate && export MUJOCO_GL=egl
cd ~/Github/H1_2_Humanoid_Control/h1_2_algoritms
R=results/bc; A="--escenario exigente --perfil espera --workers 6"
date; echo generar;  python3 demos/bc_faja.py generar  $A --episodios 40 > $R/generar_exigente_espera_log.txt 2>&1 || echo FALLO generar
date; echo entrenar; python3 demos/bc_faja.py entrenar $A --dagger 2 > $R/entrenar_exigente_espera_log.txt 2>&1 || echo FALLO entrenar
date; echo evaluar;  python3 demos/bc_faja.py evaluar  $A --episodios 90 --out evaluacion_exigente_espera.csv > $R/evaluar_exigente_espera_log.txt 2>&1 || echo FALLO evaluar
date; echo fin; tail -6 $R/evaluar_exigente_espera_log.txt
