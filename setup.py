import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
USERNAME="sathishbilli",
PROJECT_NAME="Ann_implementation"
setuptools.setup(
    name=f"{PROJECT_NAME}{USERNAME}",
    version="0.0.1",
    author=f"{USERNAME}",
    author_email="sathishmahi398@gmail.com",
    description="this a ann implementation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USERNAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USERNAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas" 
    ]
)