from setuptools import find_packages, setup

setup(
    name="modelkeeper",
    version="0.1",
    description="A Model Manager to Accelerate DNN Training via Automated Training Warmup",
    url="https://github.com/SymbioticLab/ModelKeeper",
    author="Fan Lai, Yinwei Dai",
    author_email="fedscale@googlegroups.com",
    package_dir={
        '': 'modelkeeper'},
    packages=find_packages('modelkeeper'),
)
