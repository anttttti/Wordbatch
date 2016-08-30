#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name='Wordbatch',
    version='0.9.9',
    description='Parallel text feature extraction for machine learning',
    url='https://github.com/anttttti/Wordbatch',
    author='Antti Puurula',
    author_email='antti.puurula@yahoo.com',
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
    install_requires=['scikit-learn', 'python-Levenshtein'], 
    extras_require={'dev': ['nltk', 'textblob', 'neon', 'pandas']},
    
        
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("wordbatch",
                             sources=["wordbatch.pyx", "MurmurHash3.cpp"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args = ["-ffast-math"], 
    )
    ]
    
)
