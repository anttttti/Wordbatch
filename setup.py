#!/usr/bin/env python
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name='Wordbatch',
    version='1.1.0',
    description='Parallel text feature extraction for machine learning',
    url='https://github.com/anttttti/Wordbatch',
    author='Antti Puurula',
    author_email='antti.puurula@yahoo.com',

    packages=['wordbatch',
              'wordbatch.models'
    ],    

    license='GNU GPL 2.0',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=['cython', 'scikit-learn', 'python-Levenshtein', 'py-lz4framed'], 
    extras_require={'dev': ['nltk', 'textblob', 'neon', 'pandas']},
    
        
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("wordbatch.wordbatch",
                             ["wordbatch/wordbatch.pyx", "wordbatch/MurmurHash3.cpp"],
                             libraries= [],
                             include_dirs=[numpy.get_include(), '.'],
                             extra_compile_args = ["-O3", "-ffast-math"]),
                   Extension("wordbatch.models.ftrl",
                             ["wordbatch/models/ftrl.pyx"],
                             libraries= [],
                             include_dirs=[numpy.get_include(), '.'],
                             extra_compile_args = ["-O3", "-fopenmp", "-ffast-math"],
                             extra_link_args=['-fopenmp'])
    ]
)
