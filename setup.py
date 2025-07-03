'''
The setup.py file is an essential part of packaging and distributing Python projects.
it is used by setuptools to define the configuration of the package,
including its name, version, author, description, dependencies, and other metadata.

'''

from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='Flu_Shot_Learning',
    version='0.0.1',
    author='Akajiaku',
    author_email='emmanuelalozieuwa@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)