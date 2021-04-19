from distutils.core import setup
from Cython.Build import cythonize
import glob

setup(
   ext_modules = cythonize(glob.glob('utils/*.py'))
)

# python cython_setup.py build_ext --inplace
# rm -rf utils/*.c utils/*.so