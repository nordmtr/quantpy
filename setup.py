import setuptools


with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='quantpy',
    version='0.2',
    description='Framework for quantum computations and tomography',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Dmitry Norkin, Anton Bozhedarov',
    author_email='nordmtr@gmail.com',
    install_requires=[
        'numpy',
        'scipy',
    ],
    packages=setuptools.find_packages(),
)
