import os
from glob import glob

from setuptools import setup

package_name = "h1_2_sim2real"
# comun/ (protocolo, juntas, modbus) y robot/ (agente SDK) viven fuera del paquete
# porque tambien se copian al robot, que no tiene ROS. Se instalan en share/.
RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def rel(*p):
    return os.path.relpath(os.path.join(RAIZ, *p))


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/mjcf", glob("mjcf/*.xml")),
        ("share/" + package_name + "/meshes", glob("meshes/*")),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/politicas", glob("politicas/*")),
        ("share/" + package_name + "/comun", glob(rel("comun", "*.py"))),
        ("share/" + package_name + "/robot", glob(rel("robot", "*"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MiTo Olortegui Huaman",
    maintainer_email="molortegui@utec.edu.pe",
    description="H1-2: el mismo comando ROS 2 / SDK para MuJoCo (con manos Inspire y D435) y para el robot real",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim = h1_2_sim2real.sim_node:main",
            "puente = h1_2_sim2real.puente_node:main",
        ],
    },
)
