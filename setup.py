from setuptools import setup, find_packages

with open("./README.md", "r") as f:
    long_description = f.read()

requires = [
    "click>=8.1.0",
    "jinja2>=3.1.0",
    "pyroaring>=0.3.3",
    "regex>=2022.4.24",
    "starlette>=0.20.0",
    "uvicorn>=0.17.0",
]


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
