from setuptools import find_packages, setup

setup(
    name='Symptom2Disease',
    version='0.1.0',
    description='A machine learning project to predict diseases from symptoms.',
    author='Your Name',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],
)
