from setuptools import setup

setup(name='mfact',
      version='0.1',
      url='http://github.com/yangkky/mfact',
      description='Matrix factorization models',
      packages=['mfact'],
      license='MIT',
      author='Kevin Yang',
      author_email='seinchin@gmail.com',
      test_suite='nose.collector',
      tests_require=['nose'])
