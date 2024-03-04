from setuptools import setup, find_packages

setup(
    name='super',
    version='0.0.0',
    packages=find_packages(),
    description="A command line interface",
    python_requires='>=3.6',
    install_requires=[
        'click',
    ],
    entry_points='''
        [console_scripts]
        super=supercli:supercli
    ''',
)