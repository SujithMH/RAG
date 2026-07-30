import io
import os
import fitz  # PyMuPDF
import ollama
import pdfplumber
from PIL import Image


def extract_advanced_content(pdf_path: str) -> str:
    full_content = ""

    # --- 1. EXTRACT TEXT & TABLES ---
    print("Extracting text and tables...")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_content += f"\n[Page {i+1} Text]\n{text}\n"

            tables = page.extract_tables()
            for j, table in enumerate(tables):
                full_content += f"\n[Page {i+1} Table {j+1}]\n"
                for row in table:
                    clean_row = [
                        str(cell).replace("\n", " ") if cell else ""
                        for cell in row
                    ]
                    full_content += " | ".join(clean_row) + "\n"

    # --- 2. EXTRACT & DESCRIBE IMAGES (OPTIMIZED) ---
    print("Scanning for meaningful images to describe...")
    doc = fitz.open(pdf_path)

    for i in range(len(doc)):
        page = doc[i]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            width = base_image["width"]
            height = base_image["height"]

            # OPTIMIZATION 1: Skip tiny graphics (logos, icons, line art)
            if width < 200 or height < 200:
                print(
                    f"Skipping small graphic ({width}x{height}) on page"
                    f" {i+1}..."
                )
                continue
            # Load image into PIL
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))

            # OPTIMIZATION 3: Check if image is completely solid/empty
            extrema = pil_img.convert("L").getextrema() # Convert to grayscale to check brightness
            if extrema[0] == extrema[1]:
                print(f"Skipping empty/solid color image on page {i+1}...")
                continue

            # OPTIMIZATION 2: Resize large images to max 512px for faster vision LLM inference
            pil_img.thumbnail((512, 512))

            # Load image into PIL
            temp_path = f"temp_img_{i}_{img_index}.png"
            pil_img.save(temp_path)

            print(
                f"Asking Llava to describe diagram ({width}x{height}) on page"
                f" {i+1}..."
            )

            try:
                response = ollama.chat(
                    model="llava",
                    messages=[{
                        "role": "user",
                        "content": (
                            "Explain the data, charts, or concepts shown in"
                            " this image in detail. Extract any relevant text."
                        ),
                        "images": [temp_path],
                    }],
                )
                description = response["message"]["content"]
                full_content += (
                    f"\n[Page {i+1} Image {img_index+1}"
                    f" Description]\n{description}\n"
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    return full_content


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    # The hierarchy of separators: Paragraphs -> Lines -> Sentences -> Words -> Characters
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(text_to_split: str, separator_index: int) -> list[str]:
        # Base case: if the text is already small enough, return it
        if len(text_to_split) <= chunk_size:
            return [text_to_split]
            
        separator = separators[separator_index]
        
        # If the separator isn't in the text, drop down to the next smaller separator
        if separator != "" and separator not in text_to_split:
            return split_text(text_to_split, separator_index + 1)
            
        # Split the text by the current separator
        splits = text_to_split.split(separator) if separator else list(text_to_split)
        
        good_chunks = []
        current_chunk = []
        current_length = 0
        
        for s in splits:
            # If adding this piece exceeds the limit, save the current chunk
            if current_length + len(s) + (len(separator) if current_length > 0 else 0) > chunk_size and current_length > 0:
                good_chunks.append(separator.join(current_chunk))
                
                # Overlap Management: Remove items from the beginning of current_chunk 
                # until its length is less than or equal to the desired overlap limit
                while current_length > overlap and len(current_chunk) > 1:
                    removed_item = current_chunk.pop(0)
                    current_length -= len(removed_item) + len(separator)
                    
            current_chunk.append(s)
            current_length += len(s) + (len(separator) if len(current_chunk) > 1 else 0)
            
        # Append whatever is left over
        if current_chunk:
            good_chunks.append(separator.join(current_chunk))
            
        # Final safety check: if any merged chunk is STILL too big, recursively break it down further
        final_chunks = []
        for chunk in good_chunks:
            if len(chunk) > chunk_size and separator_index < len(separators) - 1:
                final_chunks.extend(split_text(chunk, separator_index + 1))
            else:
                final_chunks.append(chunk)
                
        return final_chunks

    return split_text(text, 0)