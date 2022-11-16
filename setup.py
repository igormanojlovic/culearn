from setuptools import setup


def parse_requirements(filename='requirements.txt'):
    """ load requirements from a pip requirements file """
    lines = (line.strip() for line in open(filename))
    return [line for line in lines if line and not line.startswith("#")]


setup(
    name='culearn',
    version='1.0',
    packages=['culearn'],
    license='MIT',
    author='Igor Manojlović',
    author_email='igor.manojlovic.rs@gmail.com',
    url='https://github.com/igormanojlovic/culearn',
    description='culearn: Python Package for Cumulant Learning',
    install_requires=parse_requirements(),
)
