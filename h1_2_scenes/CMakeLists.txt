cmake_minimum_required(VERSION 3.8)
project(h1_2_scenes)

find_package(ament_cmake REQUIRED)
find_package(h1_2_description REQUIRED)

# Las escenas usan las mallas del robot (de h1_2_description) y las propias de
# este paquete (faja y mesa), pero MuJoCo solo admite un meshdir. En el
# código fuente quedan con meshdir="../meshes/" (igual que en
# h1_2_description); al instalar se reescriben con rutas absolutas:
#   meshdir                 -> share/h1_2_description/meshes/
#   mallas de este paquete  -> share/h1_2_scenes/meshes/<malla>
get_filename_component(H12_DESCRIPTION_SHARE "${h1_2_description_DIR}/.." ABSOLUTE)
set(SCENES_SHARE "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}")
set(SCENE_MESHES faja_L201.stl surgery_table.stl)

file(GLOB SCENES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/mjcf" "${CMAKE_CURRENT_SOURCE_DIR}/mjcf/*.xml")
foreach(scene ${SCENES})
  # Volver a configurar si cambia una escena
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "mjcf/${scene}")
  file(READ "mjcf/${scene}" content)
  string(REPLACE "meshdir=\"../meshes/\"" "meshdir=\"${H12_DESCRIPTION_SHARE}/meshes/\"" content "${content}")
  foreach(mesh ${SCENE_MESHES})
    string(REPLACE "file=\"${mesh}\"" "file=\"${SCENES_SHARE}/meshes/${mesh}\"" content "${content}")
  endforeach()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/mjcf/${scene}" "${content}")
endforeach()

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mjcf" meshes
  DESTINATION share/${PROJECT_NAME})

ament_package()
