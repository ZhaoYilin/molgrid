from setuptools import setup,find_packages

import os
import sys
import platform

setup(name='molgrid',
    version='0.1',
    description='MolGrid: A Python library for generating real-space finite grids for DFT calculations.',
    url='https://zhaoyilin.github.io/molgrid/',
    author='Yilin Zhao',
    author_email='zhaoyilin10@foxmail.com',
    license='MIT',
    packages=find_packages(),
    package_data={'molgrid': ['data/*']})
