import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'micoes'
AUTHOR = 'Egawati Panjei'
AUTHOR_EMAIL = 'egawati.panjei@ou.edu'
URL = 'https://github.com/egawati/micoes'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Microclusters based outlier explanations for data stream'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'sklearn'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages('src'),
      package_dir={'': 'src'},
      python_requires='>=3.6',
      )
