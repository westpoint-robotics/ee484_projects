from setuptools import find_packages
from setuptools import setup
import os
from glob import glob

package_name = 'ee484_projects'

setup(
    name=package_name,
    version='0.0.1',
    # packages=[package_name],
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name), glob('ee484_projects/*.py')), # <-- This line is crucial
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'launch'],
    zip_safe=True,
    maintainer='Dominic Larkin',
    maintainer_email='dominic.larkin@example.com',
    description='A Package that makes the OpenManipulator arm stand staight up',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'manipulator_straight = ee484_projects.manipulator_straight:main',
            'manipulator_node = ee484_projects.ee484_manipulator_node:main',
            # Add other entry points here as needed
        ],
    },
)