from setuptools import setup
from setuptools import find_packages

version = "1.0.1"

setup(
    name="connection",
    version=version,
    description="This package contains connection script used to connect to postgre database",
    keywords="",
    author="Faithful Alabi",
    author_email="faithfulalabi@gmail.com",
    license="Proprietary",
    packages=find_packages("connection"),
    zip_safe=False,
    install_requires=["psycopg2-binary"],
)
