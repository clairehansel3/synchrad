import platform
import setuptools
import subprocess

with open('README', 'r') as f:
    long_description = f.read()

if platform.system() == 'Darwin':
    libsynchrad = 'libsynchrad.dylib'
else:
    libsynchrad = 'libsynchrad.so'

subprocess.run(['mkdir', '-p', '.build'], check=True)
subprocess.run(['cmake', '-S', '.', '-B', '.build'], check=True)
subprocess.run(['cmake', '--build', '.build'], check=True)
subprocess.run(['mv', f'.build/{libsynchrad}', '.'], check=True)

setuptools.setup(
    name='synchrad',
    version='0.0.1',
    description='a python package for synchrotron radiation calculation',
    long_description=long_description,
    url='https://github.com/clairehansel3/synchrad',
    author='Claire Hansel',
    author_email='clairehansel3@gmail.com',
    license='GPLv3',
    packages=['synchrad'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    data_files=[libsynchrad]
)
