from package_settings import NAME, VERSION, PACKAGES, DESCRIPTION
from setuptools import setup


setup(
    name=NAME,
    version=VERSION,
    long_description=DESCRIPTION,
    author="Christoph Alt",
    author_email="christoph.alt@posteo.de",
    packages=PACKAGES,
    include_package_data=True,
    install_requires=["ray==0.6.2"],
    package_data={"": ["*.*"]},
)
