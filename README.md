# PDF Caption Image Extractor

A Python-based tool that automatically extracts images associated with a specific searchable caption from PDF documents.

Unlike simple PDF image extractors, this project identifies the desired caption anywhere in the document, locates the corresponding image using coordinate mapping and computer vision, and saves only the relevant image.

---

## Features

- Extract images associated with any searchable caption
- Search across every page of the PDF
- Uses PDF text coordinates instead of fixed page numbers
- Supports multiple occurrences of the same caption
- Automatically detects the corresponding image above the caption
- Combines PDF metadata with OpenCV-based image detection for higher accuracy
- Removes unnecessary white borders from extracted images
- Batch processing of multiple PDF files
- Configurable caption label through command-line arguments
- Debug mode for visualizing ROI, detected candidates, and final extraction
- Automatic Poppler detection on Windows

---

## Project Structure

```
pdf-caption-image-extractor/
│
├── pd_image_scraper.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── sample_pdfs/
│   └── sample.pdf
│
├── output/
│
└── debug/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pdf-caption-image-extractor.git

cd pdf-caption-image-extractor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Poppler

This project uses **pdf2image**, which requires Poppler.

Download Poppler for your operating system and either:

- Add its `bin` directory to your system PATH

or

- Pass the directory using:

```bash
--poppler-path
```

The script also attempts to automatically detect common Windows Poppler installations.

---

## Usage

Basic usage:

```bash
python pd_image_scraper.py \
-i input_pdfs \
-o output \
--label "Target Caption Text"
```

Example:

```bash
python pd_image_scraper.py \
-i documents \
-o extracted_images \
--label "Figure"
```

---

## Command-Line Options

| Option | Description |
|----------|-------------|
| `-i, --input` | Input folder containing PDF files |
| `-o, --output` | Output folder for extracted images |
| `--label` | Searchable caption associated with the image |
| `--dpi` | Rendering DPI (default: 250) |
| `--debug` | Save debug visualizations |
| `--debug-folder` | Directory for debug output |
| `--overwrite` | Replace existing extracted images |
| `--poppler-path` | Path to Poppler's bin directory |
| `--log-level` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## How It Works

The extraction pipeline consists of the following steps:

1. Search every page using **pdfplumber** for the specified caption.
2. Convert PDF text coordinates into rendered image coordinates.
3. Build a Region of Interest (ROI) above the detected caption.
4. Detect image candidates using OpenCV contour analysis.
5. Include embedded PDF image metadata as additional candidates.
6. Score each candidate based on:
   - Distance from the caption
   - Image size
   - Aspect ratio
   - Detection source
7. Select the highest-ranked candidate.
8. Remove unnecessary white borders.
9. Save the final extracted image.

This approach avoids relying on fixed page layouts, making it robust across different PDF formats.

---

## Technologies Used

- Python
- OpenCV
- pdfplumber
- pdf2image
- Pillow
- NumPy
- tqdm

---

## Requirements

```
pdfplumber>=0.11.0
pdf2image>=1.17.0
opencv-python>=4.9.0
numpy>=1.26.0
Pillow>=10.0.0
tqdm>=4.66.0
```

---

## Example Output

```
output/
│
├── document_1/
│   ├── target_caption_001.jpg
│   └── target_caption_002.jpg
│
├── document_2/
│   └── target_caption_001.jpg
```

Debug mode additionally produces:

```
debug/
│
├── page_roi.jpg
├── candidate_boxes.jpg
└── final_crop.jpg
```

---

## Future Improvements

- Support scanned PDFs using OCR
- Interactive desktop GUI
- Parallel PDF processing
- Automatic image quality enhancement
- Export extraction reports in CSV or JSON
- Docker support

---

## Applications

This project can be used for:

- Financial reports
- Insurance documents
- Survey reports
- Inspection reports
- Construction reports
- Medical documents
- Research papers
- Legal documents
- Any PDF containing searchable captions linked to images

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

---

## Support

If you find this project useful, please consider giving it a ⭐ on GitHub!