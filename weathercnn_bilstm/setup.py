from setuptools import setup, find_packages

setup(
    name="weathercnn_bilstm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow>=2.0.0",
        "scikit-learn",
        "matplotlib",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Weather temperature prediction using CNN-BiLSTM hybrid models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/weathercnn_bilstm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)