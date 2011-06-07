import setuptools

setuptools.setup(
    name='lmj.pursuit',
    version='0.1',
    install_requires=['numpy'],
    py_modules=['lmj.pursuit'],
    ext_modules=[setuptools.Extension('lmj._correlate', sources=['lmj/correlate.c'])],
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='A small library of matching pursuit implementations',
    long_description=open('README.rst').read(),
    license='MIT',
    keywords=('matching-pursuit '
              'sparse-coding '
              'compressed-sensing '
              'machine-learning'),
    url='http://github.com/lmjohns3/py-pursuit/',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
