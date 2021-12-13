
from setuptools import setup, find_packages
from securityaware.core.version import get_version

VERSION = get_version()

f = open('README.md', 'r')
LONG_DESCRIPTION = f.read()
f.close()

setup(
    name='securityaware',
    version=VERSION,
    description='Fine-grained approach to detect and patch vulnerabilities',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Eduard Pinconschi',
    author_email='eduard.pinconschi@tecnico.ulisboa.pt',
    url='https://github.com/johndoe/myapp/',
    license='unlicensed',
    packages=find_packages(exclude=['ez_setup', 'tests*']),
    package_data={'securityaware': ['templates/*']},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        securityaware = securityaware.main:main
    """,
)
