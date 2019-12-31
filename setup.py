#!/usr/bin/env python
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy
import os

if os.name == 'nt':
	extra_compile_args = ["/openmp", "/Ox", "/arch:AVX2", "/fp:fast"]
	extra_link_args = []
else:
	extra_compile_args = ["-O3", "-fopenmp", "-ffast-math", "-mavx2", "-march=native", "-ftree-vectorize", "-std=gnu11"]
	extra_link_args = ["-fopenmp"]

setup(
	name='Wordbatch',
	version='1.4.4',
	description='Python library for distributed AI processing pipelines, using swappable scheduler backends',
	url='https://github.com/anttttti/Wordbatch',
	author='Antti Puurula',
	author_email='antti.puurula@yahoo.com',

	packages=['wordbatch',
			  'wordbatch.pipelines',
			  'wordbatch.extractors',
			  'wordbatch.models',
			  'wordbatch.transformers'
	],

	license='GNU GPL 2.0',
	classifiers=[
		"Development Status :: 4 - Beta",
		"Intended Audience :: Developers",
		"License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Cython",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Software Development :: Libraries :: Python Modules",
	],
	install_requires=['Cython', 'scikit-learn', 'python-Levenshtein', 'py-lz4framed', 'randomgen', 'numpy', 'scipy',
						  'pandas', 'wheel>=0.33.4'],
	extras_require={'dev': ['nltk', 'textblob', 'keras', 'pyspark', 'dask', 'distributed', 'ray']},


	cmdclass= {'build_ext': build_ext},
	ext_modules= [Extension("wordbatch.extractors.extractors",
							["wordbatch/extractors/extractors.pyx", "wordbatch/extractors/MurmurHash3.cpp"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.ftrl",
							["wordbatch/models/ftrl.pyx"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.fm_ftrl",
							["wordbatch/models/fm_ftrl.pyx", "wordbatch/models/avx_ext.c"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.nn_relu_h1",
							["wordbatch/models/nn_relu_h1.pyx"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
				  Extension("wordbatch.models.nn_relu_h2",
							["wordbatch/models/nn_relu_h2.pyx"],
							libraries= [],
							include_dirs=[numpy.get_include(), '.'],
							extra_compile_args = extra_compile_args,
							extra_link_args=extra_link_args),
		]
)
