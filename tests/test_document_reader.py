import pytest
from src.core.document_reader import DocumentReader
import os

def test_read_text_file(document_reader):
    """Test reading a plain text file"""
    # Create a test text file
    test_content = "This is a test resume content."
    with open("test.txt", "w") as f:
        f.write(test_content)
    
    # Read the file
    text, used_ocr = document_reader.read_document("test.txt")
    
    # Verify results
    assert text == test_content
    assert not used_ocr
    
    # Cleanup
    os.remove("test.txt")

def test_read_nonexistent_file(document_reader):
    """Test reading a non-existent file"""
    text, used_ocr = document_reader.read_document("nonexistent.txt")
    assert text is None
    assert not used_ocr

def test_detect_encoding(document_reader):
    """Test encoding detection"""
    # Create test files with different encodings
    test_content = "Test content with special chars: é, ñ, ü"
    
    # UTF-8
    with open("test_utf8.txt", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    # Read and verify
    text, _ = document_reader.read_document("test_utf8.txt")
    assert text == test_content
    
    # Cleanup
    os.remove("test_utf8.txt")

def test_ocr_required(document_reader):
    """Test OCR requirement detection"""
    # This test would require a sample PDF or image file
    # For now, we'll mock the behavior
    assert document_reader._requires_ocr("test.pdf") is True
    assert document_reader._requires_ocr("test.txt") is False

def test_supported_formats(document_reader):
    """Test supported file format detection"""
    assert document_reader._is_supported_format("test.pdf")
    assert document_reader._is_supported_format("test.doc")
    assert document_reader._is_supported_format("test.docx")
    assert document_reader._is_supported_format("test.txt")
    assert not document_reader._is_supported_format("test.xyz")

def test_file_size_check(document_reader):
    """Test file size validation"""
    # Create a test file
    with open("test.txt", "w") as f:
        f.write("x" * 1024)  # 1KB of data
    
    # Test size check
    assert document_reader._check_file_size("test.txt")
    
    # Cleanup
    os.remove("test.txt")

def test_error_handling(document_reader):
    """Test error handling for various scenarios"""
    # Test with invalid file
    text, used_ocr = document_reader.read_document("invalid/file/path.txt")
    assert text is None
    assert not used_ocr
    
    # Test with unsupported format
    text, used_ocr = document_reader.read_document("test.xyz")
    assert text is None
    assert not used_ocr 