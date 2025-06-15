# Resume Parser System

A high-performance, offline resume parsing system that converts resumes (PDF, DOCX, DOC, TXT) into structured JSON format for ATS integration.

## Features

- Multi-format support (PDF, DOCX, DOC, TXT)
- OCR fallback for scanned documents
- Parallel processing with memory management
- Advanced NLP-based extraction
- Confidence scoring
- ATS API integration
- Comprehensive logging and metrics

## Requirements

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- Tesseract OCR
- Poppler (for PDF processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resume_parser_system
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python scripts/setup_environment.py
```

## Usage

1. Place resumes in the input directory:
```bash
data/input/
```

2. Run the parser:
```bash
python scripts/process_resumes.py --input-dir data/input --output-dir data/output
```

### Command Line Options

- `--input-dir`: Directory containing resumes (default: data/input)
- `--output-dir`: Directory for output JSON files (default: data/output)
- `--batch-size`: Number of resumes to process in each batch (default: 500)
- `--num-workers`: Number of parallel workers (default: 4)
- `--push-to-api`: Push results to ATS API (requires API configuration)

### Performance Tuning

For optimal performance on your Dell XPS 13 7390:

- 8GB RAM:
  - Batch size: 300
  - Workers: 3
  - Max memory: 75%

- 16GB RAM:
  - Batch size: 500
  - Workers: 4
  - Max memory: 80%

## Output Format

The system generates two files:

1. `resumes_YYYYMMDD_HHMMSS.json`: Contains all parsed resume data
2. `processing_summary.json`: Contains processing metrics and statistics

### Resume Data Structure

```json
{
  "first_name": "John",
  "middle_name": "",
  "last_name": "Doe",
  "primary_email": "john.doe@example.com",
  "secondary_email": "",
  "phone": "5551234567",
  "secondary_phone_number": "",
  "city": "New York",
  "state": "NY",
  "zip": "10001",
  "designation": "Software Engineer",
  "experience": "5.5",
  "skills": "python, java, javascript, react, aws",
  "work_authority": "US Citizen",
  "resume_link": "/path/to/resume.pdf",
  "raw_resume": "...",
  "source_by": "Direct",
  "tax_term": "W2",
  "payrate": "100000",
  "comment": "--",
  "added_by": "System",
  "processed_at": "2024-01-20T10:30:00",
  "confidence_score": 0.95
}
```

## Configuration

Create a `.env` file in the project root:

```env
# API Configuration
ATS_API_URL=https://api.example.com/v1
ATS_API_KEY=your_api_key

# Processing Configuration
BATCH_SIZE=500
NUM_WORKERS=4
MAX_MEMORY_PERCENT=80
ENABLE_OCR=true
OCR_CONFIDENCE_THRESHOLD=0.6
```

## Troubleshooting

1. **Out of Memory**
   - Reduce batch size and number of workers
   - Monitor memory usage in logs
   - Consider processing in smaller chunks

2. **Slow OCR**
   - Disable OCR for first pass
   - Process OCR separately for failed documents
   - Adjust OCR confidence threshold

3. **Low Accuracy**
   - Check spaCy model installation
   - Verify skill taxonomy
   - Review confidence scores

4. **API Errors**
   - Check API credentials
   - Verify network connectivity
   - Review rate limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 