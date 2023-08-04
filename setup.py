# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['instacrawl']

package_data = \
{'': ['*']}

install_requires = \
['beautifulsoup4>=4.12.2,<5.0.0',
 'dtale>=3.3.0,<4.0.0',
 'eva-decord>=0.6.1,<0.7.0',
 'face-recognition-models>=0.3.0,<0.4.0',
 'face-recognition>=1.3.0,<2.0.0',
 'html5lib>=1.1,<2.0',
 'kaleido==0.2.1',
 'numpy>=1.25.2,<2.0.0',
 'opencv-python>=4.8.0.74,<5.0.0.0',
 'pandas>=2.0.3,<3.0.0',
 'pillow>=10.0.0,<11.0.0',
 'pynput>=1.7.6,<2.0.0',
 'python-dotenv>=1.0.0,<2.0.0',
 'requests>=2.31.0,<3.0.0',
 'rich>=13.5.2,<14.0.0',
 'scikit-learn>=1.3.0,<2.0.0',
 'selenium-wire>=5.1.0,<6.0.0',
 'selenium>=4.11.2,<5.0.0',
 'timm>=0.9.2,<0.10.0',
 'torch>=2.0.1,<3.0.0',
 'tqdm>=4.65.0,<5.0.0',
 'transformers>=4.31.0,<5.0.0',
 'typer>=0.9.0,<0.10.0',
 'typing-extensions>=4.7.1,<5.0.0']

entry_points = \
{'console_scripts': ['instacrawl = instacrawl.__main__:run']}

setup_kwargs = {
    'name': 'instacrawl',
    'version': '0.1.4',
    'description': 'A simple CLI Instagram crawler with a focus on algorithm analytics.',
    'long_description': '',
    'author': 'Analetta Ehler',
    'author_email': 'annie.ehler.4@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/annie444/instacrawl',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.10,<3.13',
}


setup(**setup_kwargs)

