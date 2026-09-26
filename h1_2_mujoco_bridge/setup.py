import os
from glob import glob

from setuptools import setup

package_name = 'h1_2_mujoco_lowlevel_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='usuario',
    maintainer_email='user@example.com',
    description='Bridge ROS2-MuJoCo de bajo nivel (lowcmd/lowstate) para el H1-2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lowlevel_bridge = h1_2_mujoco_lowlevel_bridge.bridge_node:main',
            'offline_demo = h1_2_mujoco_lowlevel_bridge.offline_demo:main',
        ],
    },
)
