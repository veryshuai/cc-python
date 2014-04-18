try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Estimates optimal sales tax',
    'author': 'David Jinkins',
    'url': 'www.davidjinkins.com',
    'download_url': 'github.org/veryshuai/',
    'author_email': 'david.jinkins@gmail.com',
    'version': '0.1',
    'install_requires': ['pandas', 'numpy'],
    'name': 'sales_tax'
}

setup(**config)
