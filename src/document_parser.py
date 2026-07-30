import io
import os
import fitz 
import ollama
import pdfplumber
from PIL import Image

def extract_advanced_content(pdf_path: str) -> list[dict]:
    pages_data = []
    filename = os.path.basename(pdf_path)

    # --- 1. EXTRACT TEXT & TABLES (PAGE BY PAGE) ---
    print("Extracting text and tables...")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = ""
            text = page.extract_text()
            if text:
                page_text += f"{text}\n"

            tables = page.extract_tables()
            for j, table in enumerate(tables):
                page_text += f"\n[Table {j+1}]\n"
                for row in table:
                    clean_row = [str(cell).replace("\n", " ") if cell else "" for cell in row]
                    page_text += " | ".join(clean_row) + "\n"
                    
            # Save the page as a structured dictionary
            pages_data.append({
                "text": page_text,
                "page": i + 1,
                "source": filename
            })

    # --- 2. EXTRACT & DESCRIBE IMAGES ---
    print("Scanning for meaningful images to describe...")
    doc = fitz.open(pdf_path)

    for i in range(len(doc)):
        page = doc[i]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            width, height = base_image["width"], base_image["height"]
            if width < 200 or height < 200:
                continue

            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))

            extrema = pil_img.convert("L").getextrema() 
            if extrema[0] == extrema[1]:
                continue

            pil_img.thumbnail((512, 512))
            temp_path = f"temp_img_{i}_{img_index}.png"
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
                # Append the image description directly to the correct page dictionary
                pages_data[i]["text"] += f"\n[Image {img_index+1} Description]\n{response['message']['content']}\n"
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    return pages_data


def chunk_text(pages_data: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(text_to_split: str, separator_index: int) -> list[str]:
        if len(text_to_split) <= chunk_size:
            return [text_to_split]
            
        separator = separators[separator_index]
        if separator != "" and separator not in text_to_split:
            return split_text(text_to_split, separator_index + 1)
            
        splits = text_to_split.split(separator) if separator else list(text_to_split)
        good_chunks, current_chunk, current_length = [], [], 0
        
        for s in splits:
            if current_length + len(s) + (len(separator) if current_length > 0 else 0) > chunk_size and current_length > 0:
                good_chunks.append(separator.join(current_chunk))
                while current_length > overlap and len(current_chunk) > 1:
                    removed_item = current_chunk.pop(0)
                    current_length -= len(removed_item) + len(separator)
                    
            current_chunk.append(s)
            current_length += len(s) + (len(separator) if len(current_chunk) > 1 else 0)
            
        if current_chunk:
            good_chunks.append(separator.join(current_chunk))
            
        final_chunks = []
        for chunk in good_chunks:
            if len(chunk) > chunk_size and separator_index < len(separators) - 1:
                final_chunks.extend(split_text(chunk, separator_index + 1))
            else:
                final_chunks.append(chunk)
        return final_chunks

    # NEW LOGIC: Iterate over dictionaries instead of a single string
    chunked_data = []
    for page in pages_data:
        page_chunks = split_text(page["text"], 0)
        for chunk in page_chunks:
            if chunk.strip(): 
                chunked_data.append({
                    "text": chunk,
                    "page": page["page"],
                    "source": page["source"]
                })
                
    return chunked_data