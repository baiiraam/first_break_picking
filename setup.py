from setuptools import find_packages, setup

setup(
    name="seismic-fbp",
    version="1.0.0",
    description="Seismic First Break Picking with U-Net",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
