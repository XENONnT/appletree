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
            'flake8',
        ],
    },
    packages=setuptools.find_packages()
)
