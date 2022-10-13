import setuptools


def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires


requires = open_requirements('requirements.txt')

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(
    name='appletree',
    version='0.0.0',
    description='A high-Performance Program simuLatEs and fiTs REsponse of xEnon.',
    author='Appletree contributors, the XENON collaboration',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    setup_requires=['pytest-runner'],
    install_requires=requires,
    python_requires='>=3.8',
    extras_require={
        'doc': [],
        'test': [
            'pytest',
            'flake8',
        ],
    },
    packages=setuptools.find_packages(),
    url="https://github.com/XENONnT/appletree",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    zip_safe=False,
)
