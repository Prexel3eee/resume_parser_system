from setuptools import setup, find_packages

setup(
    name="resume_parser",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.0.0",
        "transformers>=4.0.0",
        "python-docx>=0.8.11",
        "PyPDF2>=2.0.0",
        "pytesseract>=0.3.8",
        "python-magic>=0.4.24",
        "requests>=2.25.1",
        "tqdm>=4.60.0",
        "numpy>=1.19.5",
        "pandas>=1.2.4",
        "scikit-learn>=0.24.2",
        "torch>=1.8.0",
        "python-dotenv>=0.19.0",
    ],
    python_requires=">=3.8",
) 