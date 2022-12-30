from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    packages=find_packages(exclude=['ez_setup', 'tests*']),
    package_data={'securityaware': ['templates/*']},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        securityaware = securityaware.main:main
    """,
)