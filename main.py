# Setup environment:
# The Tesseract OCR engine needs to be installed for pytesseract to work.
# https://tesseract-ocr.github.io/tessdoc/Installation.html
# To install Tesseract, first install brew:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#
# Run these commands in your terminal to add Homebrew to your PATH:
#    echo >> /Volumes/AG2/Users/agrout/.zprofile
#    echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> /Volumes/AG2/Users/agrout/.zprofile
#    eval "$(/opt/homebrew/bin/brew shellenv zsh)"
# To get started run: brew help
# Further documentation:    https://docs.brew.sh
#
# To install Tesseract using brew:
# brew install tesseract
# This formula contains only the "eng", "osd", and "snum" language data files.
# If you need any other supported languages, run `brew install tesseract-lang`.
# The tesseract directory can then be found using: brew info tesseract
# # e.g. /usr/local/Cellar/tesseract/3.05.02/share/tessdata/
#
# To install all the required packages using uv package manager :
# uv add openai pymupdf pdf2image pytesseract pillow
#
# Start local LLM server with:
# mlx_lm.server --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit --top-p 0.8 --top-k 20 --max-tokens 65536
#
# To run this python program "main.py":
# cd project_dir
# source .venv/bin/activate
# uv run main.py

import os
import sys
from datetime import datetime
import pymupdf
import pytesseract
from PIL import Image
from openai import OpenAI


# ==========================================
# 1. API CONFIGURATION
# ==========================================

# Using a custom/local OpenAI-compatible server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy-key")

# MODEL_NAME = "gpt-4o-mini"  # Change to your model name if using a local server
# MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-8bit".       # 37.7GB
# MODEL_NAME = "mlx-community/Llama-3.1-8B-Instruct"  # 16.1GB
# MODEL_NAME = "mlx-community/Qwen3-30B-A3B-8bit"  # 32.5GB
# MODEL_NAME = "mlx-community/Llama-3.3-70B-Instruct-4bit"  # 39.7GB
MODEL_NAME = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"  # 32.5GB


# ==========================================
# 2. PDF PRE-FILTERING (Your Context Search)
# ==========================================
def get_relevant_text_from_pdf(pdf_path, keywords, text_threshold=50):
    """
    Searches a PDF for keywords. If a page lacks text (scanned image),
    it automatically runs OCR. Returns the matching paragraphs to create
    the RAG "context window" for the LLM.
    """
    try:
        doc = pymupdf.open(pdf_path)
        relevant_text = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Extract standard text to check if the page is essentially empty
            page_text = page.get_text("text").strip()

            # Step A: Decide between Native Extraction or OCR
            if len(page_text) >= text_threshold:
                # Normal native extraction
                blocks = page.get_text("blocks")
            else:
                print(
                    f"    -> Page {page_num + 1}: Scanned image detected. Running OCR..."
                )
                # Render the page to an image at 300 DPI (ideal for OCR)
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Extract text using Tesseract
                ocr_text = pytesseract.image_to_string(img)

                # Create "mock blocks" by splitting paragraphs so it matches
                # the PyMuPDF block format [x0, y0, x1, y1, "text", block_no, block_type]
                paragraphs = ocr_text.split("\n\n")
                blocks = [
                    [0, 0, 0, 0, para.strip()] for para in paragraphs if para.strip()
                ]

            # Step B: Keyword Filtering (Works for both native and OCR blocks)
            for block in blocks:
                # Ensure the block has the expected structure and contains text
                if len(block) >= 5:
                    text_block = block[4].strip().lower()
                    if not text_block:
                        continue

                    # If any keyword is in this paragraph, save the original text
                    if any(keyword.lower() in text_block for keyword in keywords):
                        relevant_text.append(block[4].strip())

        doc.close()

        # Join the relevant blocks together.
        return "\n\n".join(relevant_text) if relevant_text else ""

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


# ==========================================
# 3. LLM EXTRACTION
# ==========================================
def extract_retention_time_via_llm(context_text):
    """
    Sends the filtered text to the LLM and forces a JSON response.
    """
    if not context_text:
        # Skip the API call entirely if no keywords were found!
        return "data retention time not specified"

    system_prompt = """
    You are an expert data compliance assistant. You will be provided with excerpts from a document.
    Your task is to identify the data retention time specified in the text.
    Do not show your thinking process. Respond directly with the final answer.
    
    Rules:
    1. If the retention time is found, extract it (e.g., '5 years', '30 days', 'indefinitely').
    2. If there are multiple retention times for different data, list the data types and times on separate lines
    2. If the text does not contain a specific retention time, you MUST output exactly: "data retention time not specified".
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document Excerpts:\n{context_text}"},
            ],
            temperature=0.7,
            max_tokens=65536,
        )

        # print(f"\n{response}\n\n{response.choices[0].message.content}\n\n")
        return response.choices[0].message.content

    except Exception as e:
        return f"API Error: {str(e)}"


# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def process_pdfs(base_path, file_list):
    keywords = [
        "data retention",
        "retention period",
        "kept for",
        "deleted after",
        "years",
        "months",
        "days",
    ]
    results = []

    print(f"Starting review of {len(file_list)} files...\n" + "-" * 40)

    for filename in file_list:
        file_path = os.path.join(base_path, filename)

        if not os.path.exists(file_path):
            print(f"[{filename}]: File not found at path.")
            continue

        print(f"\n\nProcessing: {filename}")

        # Step 1: Filter text locally with OCR fallback
        filtered_context = get_relevant_text_from_pdf(file_path, keywords)

        # print(f"{'*' * 40}\n{file_path}\n{filtered_context}\n\n\n")

        # Step 2: Make the API call
        retention_time = extract_retention_time_via_llm(filtered_context)

        # Step 3: Record and print
        results.append({"filename": filename, "retention_time": retention_time})
        print(f"\n\n[{filename}]:\n{'-' * 40}\n{retention_time}")

    return results


# ==========================================
# 5. TXT EXPORT
# ==========================================
def export_results_to_txt(results_data, output_filename="retention_results.txt"):
    """
    Takes the list of dictionary results and exports them to a TXT file.
    """
    if not results_data:
        print("No data found. Skipping TXT export.")
        return

    try:
        # Open the file in write mode ('w').
        # newline='' prevents blank rows between entries on Windows.
        with open(output_filename, mode="w", newline="", encoding="utf-8") as txt_file:
            for item in results_data:
                # Extract the values using the keys
                filename = item["filename"]
                raw_retention = item["retention_time"]

                # Replace the literal text '\n' with an actual Python newline character
                formatted_retention = raw_retention.replace("\\n", "\n")

                # Save the results to a file
                txt_file.write(
                    f"File: {filename}\n{'-' * 40}\n{formatted_retention}\n\n"
                )

        print(
            f"\n\nSuccessfully exported {len(results_data)} records to: {output_filename}"
        )

    except Exception as e:
        print(f"\nError saving to TXT: {e}")


def main():
    default_path = "./pdfs"
    help_flags = ["help", "-h", "-help", "--help"]

    # Check for help flags
    if len(sys.argv) > 1 and any(arg.lower() in help_flags for arg in sys.argv[1:]):
        print("\nPDF Data Retention Extractor Help\n" + "-" * 35)
        print(f"Usage: uv run {sys.argv[0]} [directory_path]")
        print("\nOptions:")
        print(
            "  [directory_path]  : The path to the folder containing PDF files to process."
        )
        print("  -h, --help        : Show this help message and exit.")
        print(f"\nDefault:")
        print(
            f"  If no path is provided, [directory_path] defaults to: '{default_path}'\n"
        )
        sys.exit(0)

    # Handle command line parameters vs defaults
    if len(sys.argv) == 1:
        common_file_path = default_path
        print(f"Note: No command line parameters provided.")
        print(f"Using default [directory_path]: {common_file_path}")
        print("To view usage instructions run with 'help' or '?'\n")
    else:
        common_file_path = sys.argv[1]

    # Validate common_file_path
    if not os.path.exists(common_file_path):
        print(f"Error: The path '{common_file_path}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(common_file_path):
        print(f"Error: The path '{common_file_path}' is not a valid directory.")
        sys.exit(1)

    # Automatically generate the list of PDF files
    pdf_files_to_check = [
        f for f in os.listdir(common_file_path) if f.lower().endswith(".pdf")
    ]

    if not pdf_files_to_check:
        print(f"Error: No .pdf files found in directory '{common_file_path}'.")
        sys.exit(1)

    # Run the processor
    final_data = process_pdfs(common_file_path, pdf_files_to_check)

    # Export the final data to a text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"document_retention_audit_{timestamp}.txt"
    export_results_to_txt(final_data, results_file)


if __name__ == "__main__":
    main()
