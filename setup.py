from setuptools import setup, find_packages

with open("./README.md", "r") as f:
    long_description = f.read()

requires = ["regex", "pyroaring", "starlette", "jinja2", "click"]

setup(
    name="hyperreal",
    use_scm_version=True,
    long_description=long_description,
    packages=find_packages("hyperreal"),
    url="https://gitlab.com/SamHames/hyperreal/",
    license="MIT",
    classifiers=[],
    entry_points={
        "console_scripts": [
            "hyperreal = hyperreal.cli:cli",
        ]
    },
    keywords="",
    install_requires=requires,
    setup_requires=["setuptools_scm"],
    author="Sam Hames",
    author_email="sam@hames.id.au",
    description="Hyperreal is a library and tool for intepretive text analytics.",
)
