from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='securityaware',
    version = "0.0.1",
    description='Fine-grained approach to detect and patch vulnerabilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['ez_setup', 'tests*']),
    package_data={'securityaware': ['templates/*']},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        securityaware = securityaware.main:main
    """,
)