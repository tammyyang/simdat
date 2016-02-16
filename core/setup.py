import glob, os
import shutil
from distutils.core import setup
from Cython.Build import cythonize

os.chdir("./")
for f in glob.glob("*.c"):
    if os.path.exists(f):
        print('Deleting %s' % f)
        os.remove(f)
    fso = os.path.join('so/', f).replace('.c', '.so')
    if os.path.exists(fso):
        print('Deleting %s' % fso)
        os.remove(fso)
setup(
    name='simdat',
    version='1.1.5',
    description='Data analysis tools.',
    author='Tammy Yang',
    ext_modules = cythonize("./*.pyx"),
)

print("Moving simdat/core/*.so to so/")
for f in glob.glob("simdat/core/*.so"):
    src = os.path.join(os.getcwd(), f)
    if os.path.exists(src):
        des = src.replace('simdat/core/simdat/core/', 'simdat/core/so/')
        shutil.move(src, des)
