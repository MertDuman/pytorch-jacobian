from setuptools import find_packages, setup

setup(
    name="pytorch_jacobian",
    packages=find_packages(),
    version="0.1.0",
    description="Various ways to calculate the Jacobian of a torch module.",
    author="Mert Duman",
    license="MIT",
    install_requires=["torch", "numpy"]
)
