import os
import numpy
from setuptools import Extension, find_packages, setup


fselect_cpp_module = Extension(
    '_fselect',
    sources=['src/fselect.cpp',
             'src/fselect_module.cpp'],
    include_dirs=[os.path.join(numpy.get_include(), 'numpy')],
    language='c++'
)

major_version = 0
minor_version = 1
subminor_version = 1
version = '%d.%d.%d' % (major_version, minor_version, subminor_version)

setup(name='bestsubset',
      python_requires='>3.5.0',
      version=version,
      description='Best subset feature selection',
      author='P. M. Larsen',
      author_email='pmla@fysik.dtu.dk',
      url='https://github.com/pmla/best-subset-selection',
      ext_modules=[fselect_cpp_module],
      install_requires=['numpy'],
      packages=['bestsubset'],
      )
