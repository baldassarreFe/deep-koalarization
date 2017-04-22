from setuptools import setup, find_packages
from pip.req import parse_requirements
from pip.download import PipSession
from carddeck import version


req_file = 'requirements.txt'
reqs = [str(r.req) for r in parse_requirements(req_file, session=PipSession())]


setup(
    name='img_transformer',
    version=version.__version__,
    install_requires=reqs,
    packages=find_packages(),
    test_suite='tests'
)
