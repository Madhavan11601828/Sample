from setuptools import setup, find_packages

setup(
    name="app-wrapper",  # Name of your package
    version="1.0.0",  # Version number
    packages=find_packages(include=["wrapper*"]),  # Include only the wrapper package
    install_requires=[
        "requests",  # Dependency for making API calls
    ],
    description="A wrapper for interacting with the deployed app",
    long_description="This library provides a wrapper to interact with the deployed application, allowing you to perform health checks and send requests for chat completions.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://your-repo-url",  # URL to your GitHub or Azure DevOps repo
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
