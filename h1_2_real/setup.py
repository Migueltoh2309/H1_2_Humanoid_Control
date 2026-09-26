from setuptools import find_packages, setup

package_name = 'h1_2_real'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='utec',
    maintainer_email='molortegui@utec.edu.pe',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_leer = h1_2_real.test_leer:main',
            'test_mandar = h1_2_real.test_mandar:main',
            'test_mandar_modificado = h1_2_real.test_mandar_modificado:main',
        ],
    },
)
