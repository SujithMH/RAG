import io
import os
import fitz 
import ollama
import pdfplumber
import tempfile
from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_advanced_content(pdf_path: str) -> list[dict]:
    filename = os.path.basename(pdf_path)
    pages_data = []

    print(f"Extracting content from {filename}...")
    
    # Open both readers simultaneously to process page-by-page
    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as doc:
        num_pages = len(doc)
        
        # Guard clause for unexpected library mismatches
        if len(pdf.pages) != num_pages:
            raise ValueError(f"Page count mismatch: pdfplumber ({len(pdf.pages)}) vs fitz ({num_pages})")

        for i in range(num_pages):
            page_text = ""
            
            # --- 1. EXTRACT TEXT & TABLES (pdfplumber) ---
            plumber_page = pdf.pages[i]
            
            text = plumber_page.extract_text()
            if text:
                page_text += f"{text}\n"

            tables = plumber_page.extract_tables()
            for j, table in enumerate(tables):
                page_text += f"\n[Table {j+1}]\n"
                for row in table:
                    clean_row = [str(cell).replace("\n", " ") if cell else "" for cell in row]
                    page_text += " | ".join(clean_row) + "\n"
                    
            # --- 2. EXTRACT & DESCRIBE IMAGES (fitz) ---
            fitz_page = doc[i]
            image_list = fitz_page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)

                width, height = base_image["width"], base_image["height"]
                if width < 200 or height < 200:
                    continue

                image_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(image_bytes))

                # Skip solid color images
                extrema = pil_img.convert("L").getextrema() 
                if extrema[0] == extrema[1]:
                    continue

                pil_img.thumbnail((512, 512))
                
                # Thread-safe temporary file generation
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                    pil_img.save(temp_path)

                try:
                    response = ollama.chat(
                        model="llava",
                        messages=[{
                            "role": "user",
                            "content": "Explain the data, charts, or concepts shown in this image in detail. Extract any relevant text.",
                            "images": [temp_path],
                        }],
                    )
                    page_text += f"\n[Image {img_index+1} Description]\n{response['message']['content']}\n"
                except Exception as e:
                    page_text += f"\n[Image {img_index+1} Processing Failed: {str(e)}]\n"
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # --- 3. SAVE PAGE DATA ---
            pages_data.append({
                "text": page_text,
                "page": i + 1,
                "source": filename
            })

    return pages_data


def chunk_text(pages_data: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False
    )

    chunked_data = []
    
    for page in pages_data:
        page_chunks = text_splitter.split_text(page["text"])
        
        for chunk in page_chunks:
            if chunk.strip(): 
                chunked_data.append({
                    "text": chunk,
                    "page": page["page"],
                    "source": page["source"]
                })
                
    return chunked_data