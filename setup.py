from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'fast log',
  ext_modules = cythonize("log_non_zero.pyx"),
)
