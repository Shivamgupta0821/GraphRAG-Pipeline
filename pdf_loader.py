import os
from pypdf import PdfReader

def load_pdfs_from_folder(folder_path: str = "pdfs") -> str:
    """
    Reads all PDFs from the given folder.
    Returns all extracted text as one combined string.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f" Created folder: {folder_path}")
        print(f" Drop your PDF files into the '{folder_path}' folder and run again.")
        return ""

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f" No PDF files found in '{folder_path}' folder.")
        print(f" Drop your PDF files there and run again.")
        return ""

    print(f"\n Found {len(pdf_files)} PDF(s):\n")

    all_text = ""

    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        print(f"  Reading: {filename}")

        try:
            reader = PdfReader(filepath)
            pdf_text = ""

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"

            print(f"  Pages read: {len(reader.pages)}")
            print(f"  Characters extracted: {len(pdf_text)}\n")

            all_text += f"\n--- Source: {filename} ---\n"
            all_text += pdf_text

        except Exception as e:
            print(f"  Could not read {filename}: {e}\n")
            continue

    return all_text


def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text.
    Removes weird characters, extra spaces, and very short lines
    that are usually headers/footers/page numbers.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        # Skip very short lines (page numbers, headers)
        if len(line) < 20:
            continue

        # Skip lines that are mostly numbers (tables of contents etc.)
        if sum(c.isdigit() for c in line) > len(line) * 0.5:
            continue

        cleaned.append(line)

    return " ".join(cleaned)