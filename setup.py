from setuptools import setup


setup(
    name="tuna",
    version="0.0.1",
    long_description="Hyperparameter tuning for AllenNLP, powered by Ray.",
    author="Christoph Alt",
    author_email="christoph.alt@posteo.de",
    include_package_data=True,
    install_requires=["ray==0.6.2"],
    package_data={"": ["*.*"]},
)
