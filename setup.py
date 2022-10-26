from setuptools import setup, find_packages

with open("./README.md", "r") as f:
    long_description = f.read()

requires = [
    "click>=8.1.0",
    "jinja2>=3.1.0",
    "pyroaring>=0.3.3",
    "regex>=2022.4.24",
    "cherrypy>=18.6.0",
    "python-dateutil>=2.8.0",
]

extras = {"test": ["pytest", "black", "tox"]}


setup(
    name="hyperreal",
    use_scm_version=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("hyperreal"),
    url="https://github.com/SamHames/hyperreal/",
    license="Apache License 2.0",
    entry_points={
        "console_scripts": [
            "hyperreal = hyperreal.cli:cli",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
    ],
    keywords="",
    install_requires=requires,
    setup_requires=["setuptools_scm"],
    extras_require=extras,
    python_requires=">=3.9.0",
    author="Sam Hames",
    description="Hyperreal is a library and tool for intepretive topic modelling.",
)
