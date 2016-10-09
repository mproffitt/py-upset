__author__ = 'leo@opensignal.com'
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='PyUpSet',
    version='0.2',
    description='Python implementation of the UpSet visualisation suite by Lex et al.',
    author = [
        'Leonardo Baldassini',
        'Martin Proffitt'
    ],
    author_email= [
        'leo@opensignal.com',
        'mproffitt@jitsc.co.uk'
    ],
    url='https://github.com/ImSoErgodic/py-upset',
    license='MIT',
    classifiers= [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 3'
    ],
    install_requires=[
        'pandas',
        'matplotlib',
        'numpy'
    ],
    packages=['pyupset'],
    package_dir={
        'pyupset': 'src/pyupset'
    },
    package_data={
        'pyupset': ['data/*.pckl']
    },
    test_suite='nose.collector',
    tests_require=[
        'nose',
        'coverage',
        'pbr',
        'mock',
        'ddt'
    ]
)
