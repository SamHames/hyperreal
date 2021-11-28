from setuptools import setup, find_packages

with open("./README.md", "r") as f:
    long_description = f.read()

requires = ["regex", "pyroaring"]

setup(
    name="hyperreal",
    use_scm_version=True,
    long_description=long_description,
    packages=find_packages("hyperreal"),
    # package_dir={"": "src"},
    url="https://gitlab.com/SamHames/hyperreal/",
    license="MIT",
    classifiers=[],
    # entry_points={"console_scripts": ["simple_journal = simple_journal.app:main"]},
    keywords="",
    install_requires=requires,
    setup_requires=["setuptools_scm"],
    extras_require={"test": ["requests"]},
    author="Sam Hames",
    author_email="sam@hames.id.au",
    description="Hyperreal is a library and tool for intepretive text analytics.",
)
