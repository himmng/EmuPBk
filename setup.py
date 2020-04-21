#!/usr/bin/env python
# encoding: utf-8

import sys
import os
from setuptools import find_packages

try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    from distutils.core import setup, Extension
    setup, Extension

from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


desc = open("README.md").read()
required = ["numpy", "emcee","tensorflow","cosmoHammer","chainconsumer","matplotlib","seaborn"]
test_requires = ["mock"]

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name="EmuPBk",
    version='0.0.1',
    author='Himanshu Tiwari',
    author_email="himanshuhimang@gmail.com",
    url="http://himmng.github.io/EmuPBk",
    license="MIT",
    packages=find_packages(PACKAGE_PATH, "Tests"),
    description="ANN based 21-cm Powespectrum and Bispectrum Emulator",
    test_requires=test_requires,
    package_data={'EmuPBk': ['data/*.dat']},
    include_package_data=True,
    keywords=["ANN based 21-cm signal EmuPBk",
              "21-cm powerspectrum and Bispectrum EmuPBk",
              "parameter estimation",
              "cosmology",
              "MCMC"],
    cmdclass = {'test': PyTest},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
)