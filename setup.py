import setuptools

setuptools.setup(
    name='lmj.pursuit',
    version='0.3',
    install_requires=['numpy'],
    namespace_packages=['lmj'],
    packages=setuptools.find_packages(),
    ext_modules=[setuptools.Extension('lmj.pursuit._correlate', sources=['lmj/pursuit/correlate.c'])],
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='A library of matching pursuit implementations',
    long_description=open('README.rst').read(),
    license='MIT',
    keywords=('matching-pursuit '
              'sparse-coding '
              'compressed-sensing '
              'machine-learning'),
    url='http://github.com/lmjohns3/py-pursuit',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
