import os
import setuptools

required = ["numpy", "emcee","tensorflow","cosmoHammer","chainconsumer","matplotlib","seaborn"]


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="EmuPBk",
    version='1.0.0',
    author='Himanshu Tiwari',
    author_email="himanshuhimang@gmail.com",
    packages=setuptools.find_packages(),
    url="http://himmng.github.io/EmuPBk",
    license="MIT License",
    description="ANN based 21-cm Powespectrum and Bispectrum Emulator",
    include_package_data=True,
    keywords=["ANN based 21-cm signal EmuPBk",
              "21-cm powerspectrum and Bispectrum EmuPBk",
              "parameter estimation",
              "cosmology",
              "MCMC"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
)
