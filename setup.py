import os
import setuptools
import numpy as np

setuptools.setup(
    name='lmj.pursuit',
    version='0.3.1',
    namespace_packages=['lmj'],
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='A library of matching pursuit implementations',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='http://github.com/lmjohns3/py-pursuit',
    keywords=('matching-pursuit '
              'sparse-coding '
              'compressed-sensing '
              'machine-learning'),
    install_requires=['numpy'],
    include_dirs=[np.get_include()],
    ext_modules=[setuptools.Extension('lmj.pursuit._correlate', sources=['lmj/pursuit/correlate.c'])],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
