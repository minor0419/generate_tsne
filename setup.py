from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("hlac_numpy", sources=["hlac_numpy.pyx"], include_dirs=['.', get_include()])
setup(name="hlac_numpy", ext_modules=cythonize([ext]))
