from setuptools import setup, find_packages

setup(
    name="besser-nn-migration",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["ast_parser_nn", "transform_code"],
)