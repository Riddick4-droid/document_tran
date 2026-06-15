from setuptools import find_packages, setup

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="rifd",
    version="0.1.0",
    description="Read Info From Documents – Layout‑aware Transformer for document understanding",
    author="Riddick Mensah",
    author_email="riddick.mensah@yahoo.com",
    url="https://github.com/yourusername/rifd",  # optional
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "rifd = main:main",   # main.py must have a main() function
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)