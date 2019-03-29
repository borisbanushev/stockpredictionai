# encoding: utf-8
from setuptools import setup, find_packages

import time_series_modelling

setup(name='time_series_modelling',
      version=time_series_modelling.__version__,
      packages=find_packages(exclude=["imgs"]),
      author='miao.lin',
      python_requires='>=3.6',
      platforms='any',
      install_requires=[
          "pandas>=0.24.2",
          "numpy",
          "scikit-learn",
          "xgboost",
          "matplotlib",
          "mxnet",
          "seaborn",
          "statsmodels"
      ]
      )