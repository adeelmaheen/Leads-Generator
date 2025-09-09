from setuptools import setup, find_packages

setup(
    name="leadfinder-ai",
    version="1.0.0",
    description="AI-powered lead finder using Exa search and OpenAI",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "exa-py>=1.0.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/leadfinder-ai",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

