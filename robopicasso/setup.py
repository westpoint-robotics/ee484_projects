from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robopicasso'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install SVG files into share/<package_name>/svg_files
        (os.path.join('share', package_name, 'svg_files'), glob(os.path.join('robopicasso','svg_files', '*.svg'))),
        # ----------------------
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tbot01',
    maintainer_email='you@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'integrated_manipulator_node = robopicasso.integrated_manipulator_node:main',
            'manipulator_node_with_kinematics = robopicasso.manipulator_node_with_kinematics:main',
            'IntegratedManipulatorNode = robopicasso.IntegratedManipulatorNode:main',
        ],
    },
)

