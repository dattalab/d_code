#!/usr/bin/env python

from distutils.core import setup, find_packages

setup(name='CellPickerpacage',
      version='1.0',
      description='Cell Picker GUI',
      author='Andrew, Konti',
      author_email='andrew_giessel@hms.harvard.edu, akontih@bu.edu',
      url='https://github.com/dattalab/dattacode/tree/master/imaging/segmentation',
      packages=find_packages(),
      install_requires=[
       "PySide >= 1.1.1",
       "numpy >= 1.7.1",
       "matplotlib >= 1.2.1",
       "scipy >= 0.12.0",
       "pymorph >= 0.96",
       "mahotas >= 1.0",
       "sklearn >= 0.13.1",
      ],
     )