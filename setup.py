import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="EmuPBk",
    version='6.5.2',
    author='Himanshu Tiwari',
    author_email="himanshuhimang@gmail.com",
    packages=setuptools.find_packages(),
    package_data={'': ['']},
    install_requires=["tensorflow", "chainconsumer", "cosmohammer", "imageio", "celluloid"],
    url="http://github.com/EmuPBk",
    license="MIT License",
    description="ANN based 21-cm Powespectrum and Bispectrum Emulator",
    include_package_data=True,
    keywords=["EmuPBk",
              "ANN emulation on 21-cm powerspectrum and Bispectrum EmuPBk",
              "parameter estimation",
              "cosmology",
              "MCMC"],
    project_urls={
        "Documentation": "https://emupbk.readthedocs.io/en/latest/"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
)
