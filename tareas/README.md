# Tareas del H1-2

Cada tarea (reto) en su carpeta, con el mismo esquema: el código corre igual en el simulador
(`../h1_2_sim2real`) y en el robot real, porque usa el SDK de Unitree en los dos.
Entorno: `source ../h1_2_sim2real/entorno.sh`. El `COLCON_IGNORE` evita que el `colcon build`
de `humanoid_ws` busque paquetes aquí.

| Tarea | Enunciado | Estado |
|---|---|---|
| [`seguidor_linea/`](seguidor_linea/README.md) | `../Reto_Seguidor_Linea_H1_2_Robotics40.pdf` (R40-RT-H1_2-0001) | Completo en simulación (4 niveles, 12/12 tiradas); sin probar en el robot |
