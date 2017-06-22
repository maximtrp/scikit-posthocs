from setuptools import setup

setup(name='posthocs',
      version='0.0.1',
      description='Statistical post-hoc analysis algorithms',
      url='http://github.com/maximtrp/posthocs',
      author='Maksim Terpilowski',
      author_email='maximtrp@gmail.com',
      license='GPLv3+',
      packages=['posthocs'],
      keywords='statistics posthoc',
      install_requires=['numpy', 'scipy', 'statsmodels', 'pandas'],
	  classifiers=[
		'Development Status :: 2 - Pre-Alpha',

		'Intended Audience :: Education',
		'Intended Audience :: Information Technology',
		'Intended Audience :: Science/Research',

		'Topic :: Scientific/Engineering :: Information Analysis',
		'Topic :: Scientific/Engineering :: Mathematics',

		'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.1',
		'Programming Language :: Python :: 3.2',
		'Programming Language :: Python :: 3.3',
		'Programming Language :: Python :: 3.4',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
	  ],
      zip_safe=False)
