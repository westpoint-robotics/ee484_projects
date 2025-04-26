from setuptools import find_packages, setup
import os, glob

package_name = 'tbot3_manipulation_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        ('share/' + package_name + '/config', glob.glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='dominic.larkin@westpoint.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'manipulatorpy_node = tbot3_manipulation_python.manipulatorpy_node:main',
            'waypoint_node = tbot3_manipulation_python.waypoint_follower:main',
            'tf2pose_node = tbot3_manipulation_python.tf2pose:main'
        ],
    },
)
