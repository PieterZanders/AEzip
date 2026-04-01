from setuptools import setup, find_packages

setup(
    name="aezip",
    version="0.1.0",
    description="Autoencoder-based MD trajectory compression",
    packages=find_packages(),
    package_data={
        "aezip": [
            "config/*.json",
            "dat/*.json",
            "dat/*.in",
        ],
    },
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "torch",
        "mdtraj",
        "scikit-learn",
        "mlcolvar",
    ],
    entry_points={
        "console_scripts": [
            "aezip-compress=aezip.compress:main",
            "aezip-decompress=aezip.decompress:main",
        ],
    },
)
