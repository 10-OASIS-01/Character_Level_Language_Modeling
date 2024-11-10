# setup.py

from setuptools import setup, find_packages

setup(
    name='LanguageModeling',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tensorboard',
    ],
)
