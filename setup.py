from setuptools import setup, find_packages

setup(
    name="book-recommender-tg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "scipy>=1.7.0",
        "annoy>=1.17.0",
        "rapidfuzz>=2.0.0",
    ],
    python_requires=">=3.8",
    author="neecat",
    author_email="horrorbaku14@gmail.com",
    description="A machine learning-based book recommendation system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/book-recommender-tg",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 