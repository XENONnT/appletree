import setuptools


def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split("/")[-1] if r.startswith("git+") else r
            for r in f.read().splitlines()
            if not r.startswith("-")
        ]
    return requires


requires = open_requirements("requirements.txt")

with open("README.md") as file:
    readme = file.read()

with open("HISTORY.md") as file:
    history = file.read()

setuptools.setup(
    name="appletree",
    version="0.4.0",
    description="A high-Performance Program simuLatEs and fiTs REsponse of xEnon.",
    author="Appletree contributors, the XENON collaboration",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    setup_requires=["pytest-runner"],
    install_requires=requires,
    python_requires=">=3.8",
    extras_require={
        "cpu": [
            # pip install appletree[cpu]
            "jax[cpu]",
        ],
        "cuda112": [
            # pip install appletree[cuda112] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # noqa: E501
            "jax[cuda]==0.3.15",
        ],
        "cuda121": [
            # pip install appletree[cuda121] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # noqa: E501
            # according to this commit, jax[cuda12_pip]==0.4.10 is valid for CUDA Toolkit 12.1
            "jax[cuda12_pip]",
        ],
    },
    packages=setuptools.find_packages(),
    package_data={
        "appletree": [
            "instructs/*",
            "parameters/*",
            "maps/*",
            "data/*",
        ],
    },
    url="https://github.com/XENONnT/appletree",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    zip_safe=False,
)
