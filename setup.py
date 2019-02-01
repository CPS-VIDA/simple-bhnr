import os

from setuptools import setup, find_packages

NAME = "safe-rl"
DESCRIPTION = "A collection of Safe RL experiments that uses ideas from Formal Methods, like Temporal logics, Barrier certificates, Bisimulation functions, etc."
URL = ""
EMAIL = "anandbal@usc.edu"
AUTHOR = "Anand Balakrishnan"

REQUIRES_PYTHON = '>=3.6'
VERSION = '0.1.0'
REQUIRED_PKGS = [
    "gym",
    "torch",
    "torchvision",
    "sympy",
    'temporal_logic @ https://github.com/anand-bala/tl-py/archive/master.zip',
    'cloudpickle',
]


EXTRAS = {
    'third-party': [
        'pybulletgym @ https://github.com/benelot/pybullet-gym/archive/master.zip',
        'Box2D'
    ],
}

EXTENSIONS = []

HERE = os.path.abspath(os.path.dirname(__file__))

ABOUT = dict()
ABOUT['__version__'] = VERSION

setup(
    name=NAME,
    version=ABOUT['__version__'],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests', 'scripts')),

    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS,
    setup_requires=["pytest-runner", "pytest-xdist"],
    tests_require=["pytest"],

    # ext_modules=cythonize(EXTENSIONS),
)
