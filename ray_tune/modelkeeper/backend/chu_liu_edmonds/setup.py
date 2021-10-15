import os
import setuptools
import sys

with open('README.md') as file:
    readme = file.read()

extra_link_args = []
extra_compile_args = ['-std=c++11', '-w']
if sys.platform == "darwin":
    extra_compile_args += ['-stdlib=libc++']
    extra_link_args += ['-stdlib=libc++']

setuptools.setup(
    name             = 'modelkeeper_backend.chu_liu_edmonds',
    version          = '1.0.1',
    description      = 'Bindings to Chu-Liu-Edmonds algorithm from TurboParser',
    long_description = readme, long_description_content_type = "text/markdown",
    author           = 'Milan Straka',
    author_email     = 'straka@ufal.mff.cuni.cz',
    url              = 'https://github.com/ufal/chu_liu_edmonds',
    license          = 'GPLv3',
    packages         = ['modelkeeper_backend'],
    ext_modules      = [setuptools.Extension(
        'modelkeeper_backend.chu_liu_edmonds',
        ['chu_liu_edmonds.pyx', 'chu_liu_edmonds_internal.cpp'],
        language = 'c++',
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args)],
    classifiers      = [
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries'
    ]
)

