from setuptools import setup, find_packages

setup(
    name='bowdpr',
    version='0.0.1',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'torch',
        'transformers',
        'datasets',
        'faiss',
        'tqdm',
        'fire',
        'pandas',
        'regex',
        'sentence_transformers',
    ],
)
