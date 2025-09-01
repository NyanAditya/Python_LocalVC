# --- START OF FILE streamlit_app.py ---

import os
import re
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
import torch
from nltk.stem import PorterStemmer
import nltk
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import traceback
import tempfile
import io # Needed for handling image bytes

# --- Download NLTK data ---
# Attempt to download quietly, show warning if fails
@st.cache_resource
def download_nltk_punkt():
    try:
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        print(f"Warning: Could not download NLTK punkt resource automatically. Error: {e}")
        return False

nltk_punkt_available = download_nltk_punkt()
if not nltk_punkt_available:
    st.warning("NLTK 'punkt' resource could not be downloaded. Sentence tokenization might fall back to a simpler method.", icon="⚠️")

# --- Configuration ---

# Set tesseract path
@st.cache_data # Cache the result of finding the path
def find_tesseract_path():
    tesseract_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
    for path in tesseract_paths:
        if os.path.exists(path):
            print(f"Using Tesseract at: {path}")
            return path
    print("Warning: Tesseract executable not found in common paths. OCR might fail.")
    return None

tesseract_cmd_path = find_tesseract_path()
if tesseract_cmd_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
else:
    st.warning("Tesseract executable not found. OCR functionality will be disabled.", icon="⚠️")

# --- Model and Embedder Loading (Cached) ---
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure OCR options (will be controlled by checkbox)
OCR_RESOLUTION = 300
# Global variable to track OCR state within functions, controlled by checkbox
USE_IMAGE_OCR = False

@st.cache_resource
def load_llm_model(model_name):
    """Loads the LLM model and tokenizer."""
    try:
        print(f"Loading LLM model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        print("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Fatal Error: Could not load LLM model '{model_name}'. Check model name, network connection, and available memory. Error: {e}", icon="🔥")
        # traceback.print_exc() # Print detailed traceback to console
        return None, None

@st.cache_resource
def load_embedder(model_name='all-MiniLM-L6-v2'):
    """Loads the Sentence Transformer embedder."""
    try:
        print(f"Loading Sentence Embedder: {model_name}...")
        embedder = SentenceTransformer(model_name)
        print("Embedder loaded successfully!")
        return embedder
    except Exception as e:
        st.error(f"Fatal Error: Could not load Sentence Transformer '{model_name}'. Error: {e}", icon="🔥")
        # traceback.print_exc()
        return None

# --- Select Model ---
# Allow model selection if desired, otherwise hardcode
# model_name = st.selectbox("Select Model", ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-8B-Instruct"]) # Example if you want selection
# Using the 3B model as per your latest code provided
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Load models and tokenizer
tokenizer, model = load_llm_model(model_name)
embedder = load_embedder()
stemmer = PorterStemmer() # No need to cache, relatively lightweight

# Exit if models failed to load
if not tokenizer or not model or not embedder:
    st.stop()

# --- Math Symbols ---
MATH_SYMBOLS = {
    '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt', '∞': '\\infty',
    '≠': '\\neq', '≤': '\\leq', '≥': '\\geq', '±': '\\pm', '→': '\\to',
    '∂': '\\partial', '∇': '\\nabla', 'π': '\\pi', 'θ': '\\theta',
    'λ': '\\lambda', 'μ': '\\mu', 'σ': '\\sigma', 'ω': '\\omega',
    'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon'
}

# --- Utility Functions (Copied from main.py, ensure `global USE_IMAGE_OCR` is present) ---

def get_stemmed_key(sentence, num_words=5):
    """Create a stemmed key for basic duplicate detection (fallback)."""
    words = re.findall(r'\w+', sentence.lower())[:num_words]
    return ' '.join([stemmer.stem(word) for word in words])

def complete_sentence(fragment):
    """Complete sentence fragments using the loaded LLM."""
    # Added check for loaded model
    if not model or not tokenizer:
        print("Warning: LLM not loaded, cannot complete sentence.")
        return fragment + "." # Basic fallback

    prompt = f"Complete this sentence to make it grammatically correct and meaningful. Keep it concise. Fragment: '{fragment}'\nCompleted sentence:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        completed_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completed_part = completed_full[len(prompt):].strip()
        if completed_part and len(completed_part) > 2:
            if re.match(r'^[A-Za-z0-9\s]', completed_part):
                completed_part = re.sub(r'<\|eot_id\|>$', '', completed_part).strip()
                # Ensure punctuation if missing
                if completed_part[-1].isalnum():
                     completed_part += '.'
                return completed_part
            else:
                 print(f"Warning: Unusual start to completed sentence: '{completed_part}'. Returning fragment.")
                 return fragment + "."
        else:
            print(f"Warning: LLM completion failed/empty for fragment: '{fragment}'. Returning fragment.")
            return fragment + "."
    except Exception as e:
        print(f"Error during sentence completion for '{fragment}': {e}")
        return fragment + "."


def preprocess_image_for_math_ocr(image):
    if image.mode != 'L':
        image = image.convert('L')
    image_array = np.array(image)
    threshold = np.mean(image_array) * 0.9
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def extract_text_from_pdf(pdf_path, detect_math=True):
    """Enhanced text extraction using PyMuPDF (fitz)."""
    global USE_IMAGE_OCR # Essential for modifying the global var based on checkbox

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"Extracting text from {pdf_path}...")
    extracted_text = ""
    page_messages = [] # Collect messages for Streamlit status

    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        for page_num, page in enumerate(doc):
            page_msg = f"Processing page {page_num+1}/{num_pages}..."
            print(page_msg) # Log to console
            page_messages.append(page_msg) # Add to list for UI update

            # --- Update Streamlit status periodically ---
            if page_num % 5 == 0: # Update every 5 pages
                 st.session_state['progress_message'] = f"Extracting Text...\n" + "\n".join(page_messages[-5:]) # Show last 5 messages
            # ---------------------------------------------

            text = page.get_text("text")
            if text:
                extracted_text += text + "\n"

            # Check global USE_IMAGE_OCR flag set by the checkbox
            if USE_IMAGE_OCR and page.get_images(full=True):
                ocr_msg = f"  - Page {page_num+1}: Found images, performing OCR..."
                print(ocr_msg)
                page_messages.append(ocr_msg)
                st.session_state['progress_message'] = f"Extracting Text...\n" + "\n".join(page_messages[-5:])

                # Ensure Tesseract is available before proceeding
                if not tesseract_cmd_path:
                     ocr_err_msg = "  - Skipping OCR: Tesseract path not configured."
                     print(ocr_err_msg)
                     page_messages.append(ocr_err_msg)
                     USE_IMAGE_OCR = False # Disable for rest of extraction if not found here
                     continue

                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_image = Image.open(io.BytesIO(image_bytes))

                        if detect_math:
                            processed_pil_image = preprocess_image_for_math_ocr(pil_image)
                        else:
                            processed_pil_image = pil_image

                        ocr_text = pytesseract.image_to_string(processed_pil_image, config='--psm 6 --oem 3')

                        if ocr_text.strip():
                            if detect_math:
                                for symbol, latex in MATH_SYMBOLS.items():
                                    ocr_text = ocr_text.replace(symbol, f" {latex} ")
                            extracted_text += " [OCR_IMG] " + ocr_text.strip() + " [/OCR_IMG]\n"
                            img_ocr_msg = f"    - OCR'd text from image {img_index} on page {page_num+1}."
                            print(img_ocr_msg)
                            # page_messages.append(img_ocr_msg) # Maybe too verbose for UI

                    except pytesseract.TesseractNotFoundError:
                         tess_err_msg = "Error: Tesseract not found during OCR. Disabling further OCR."
                         print(tess_err_msg)
                         page_messages.append(tess_err_msg)
                         st.warning(tess_err_msg, icon="⚠️")
                         USE_IMAGE_OCR = False # Disable globally
                         break # Stop processing images for this page
                    except Exception as e:
                        img_err_msg = f"Error processing image {img_index} on page {page_num+1}: {str(e)}"
                        print(img_err_msg)
                        page_messages.append(img_err_msg)
                        # Continue with next image

        doc.close()

        # Basic header/footer removal
        lines = extracted_text.split('\n')
        if len(lines) > 2:
             if len(lines[0].strip()) < 15 and re.match(r'^[\s\d\W]*$', lines[0].strip()):
                 lines = lines[1:]
             if len(lines) > 1 and len(lines[-1].strip()) < 15 and re.match(r'^[\s\d\W]*$', lines[-1].strip()):
                 lines = lines[:-1]
        extracted_text = '\n'.join(lines)

        extract_final_msg = f"Extracted {len(extracted_text)} characters."
        print(extract_final_msg)
        st.session_state['progress_message'] = extract_final_msg # Final extraction message
        return extracted_text

    except Exception as e:
        err_msg = f"Text extraction failed: {str(e)}"
        print(err_msg)
        traceback.print_exc() # Print details to console
        st.error(f"Error during PDF text extraction: {e}", icon="❌") # Show error in UI
        return "" # Return empty string on failure

def detect_math_content(text):
    # (Copy function content from main.py - no changes needed)
    math_keywords = [
        r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|derivative',
        r'function|integral|vector|matrix|variable|constant|graph|plot)\b'
    ]
    math_symbols = r'[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε]'
    function_notation = r'\b[a-zA-Z]\s?\([a-zA-Z0-9,\s]+\)'
    latex_delimiters = r'\$.*?\$|\\\(.*?\\\)|\\[a-zA-Z]+(\{.*?\})*'
    patterns = math_keywords + [math_symbols, function_notation, latex_delimiters]
    if re.search('|'.join(math_keywords), text, re.IGNORECASE):
        print("Math content detected (Keywords)")
        return True
    search_limit = 50000
    text_sample = text[:search_limit]
    for pattern in [math_symbols, function_notation, latex_delimiters]:
        if re.search(pattern, text_sample):
            print(f"Math content detected (Pattern: {pattern[:20]}...)")
            return True
    print("No significant math content detected.")
    return False

def clean_text(text):
    # (Copy function content from main.py - no changes needed)
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\b[^\s\w]{1,2}\b', '', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()

def split_text_into_chunks(text, chunk_size=800, overlap=50):
    # (Copy function content from main.py - minor print change)
    try:
        sentences = nltk.sent_tokenize(text)
        print(f"Tokenized into {len(sentences)} sentences.")
    except LookupError:
        print("Warning: NLTK 'punkt' resource not found. Falling back to simple regex splitting.")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences: sentences = [text]
    except Exception as e:
         print(f"Warning: NLTK sentence tokenizer failed: {e}. Using simple splitting.")
         sentences = re.split(r'(?<=[.!?])\s+', text)
         if not sentences: sentences = [text]

    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        sent_length = len(sentence.split())
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_words = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                overlap_words += len(s.split())
                overlap_sentences.insert(0, s)
                if overlap_words >= overlap: break
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sent_length
    if current_chunk: chunks.append(' '.join(current_chunk))

    # Optional merging and limiting chunks
    refined_chunks = []
    i = 0
    while i < len(chunks):
        # Merge small chunks (increased threshold slightly)
        if len(chunks[i].split()) < chunk_size * 0.3 and i + 1 < len(chunks):
            refined_chunks.append(chunks[i] + " " + chunks[i+1])
            i += 2
        else:
            refined_chunks.append(chunks[i])
            i += 1
    chunks = refined_chunks

    MAX_CHUNKS = 20
    if len(chunks) > MAX_CHUNKS:
        print(f"Warning: Reducing number of chunks from {len(chunks)} to {MAX_CHUNKS}")
        step = max(1, len(chunks) // MAX_CHUNKS) # Ensure step is at least 1
        chunks = [chunks[i] for i in range(0, len(chunks), step)][:MAX_CHUNKS]

    chunk_msg = f"Split text into {len(chunks)} chunks (target size: ~{chunk_size} words, overlap: {overlap} words)"
    print(chunk_msg)
    st.session_state['progress_message'] = chunk_msg # Update status
    return chunks


def determine_reading_level(grade):
    # (Copy function content from main.py - no changes needed)
    if not isinstance(grade, int) or not (1 <= grade <= 12):
        print(f"Warning: Invalid grade '{grade}'. Defaulting to middle school (grade 6).")
        grade = 6
    age = grade + 5
    if 1 <= grade <= 3: level, description = "lower", f"early elementary (grades {grade}, approx age {age}-{age+1})"
    elif 4 <= grade <= 6: level, description = "middle", f"late elementary / middle school (grades {grade}, approx age {age}-{age+1})"
    elif 7 <= grade <= 9: level, description = "higher", f"junior high / early high school (grades {grade}, approx age {age}-{age+1})"
    else: level, description = "higher", f"high school (grades {grade}, approx age {age}-{age+1})"
    return level, description

# --- Enhanced Summary Generation (Prompts and Functions) ---
prompts = {
    "lower": {
        "standard": ("..."), # Copy from main.py
        "math": ("...") # Copy from main.py
        },
    "middle": {
        "standard": ("..."), # Copy from main.py
        "math": ("...") # Copy from main.py
        },
    "higher": {
        "standard": ("..."), # Copy from main.py
        "math": ("...") # Copy from main.py
        }
}
# --- Copy the exact prompt dictionaries from your final main.py ---
prompts = {
    "lower": { # Grades 1-3
        "standard": (
            "You are summarizing text for a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Explain the main ideas using VERY simple words and short sentences.\n"
            "2. Each sentence should start on a new line with a hyphen '- '.\n"
            "3. Focus on the most basic concepts. Avoid complex details.\n"
            "4. Group similar ideas together if possible.\n"
            "5. Include ONE simple, fun activity suggestion related to the text at the end, under '## Fun Activity'.\n"
            "6. Start the summary with the heading '# Simple Summary'.\n"
            "7. Do not repeat information.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining a math topic to a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Use very simple words and analogies (like comparing numbers to toys).\n"
            "2. Break down steps simply: 'First, you do this. Next, you do that.'\n"
            "3. Use examples with small numbers or simple shapes.\n"
            "4. Include ONE simple practice question or drawing activity at the end under '## Practice Time'.\n"
            "5. Start the explanation with the heading '# Math Fun'.\n"
            "6. Format explanations clearly, maybe using bullet points '- ' for steps or examples.\n"
            "Text to explain:\n{text}"
        )
    },
    "middle": { # Grades 4-6
        "standard": (
            "You are summarizing text for a student in grades 4-6 (ages 9-11).\n"
            "Instructions:\n"
            "1. Identify the main topics and summarize the key information for each.\n"
            "2. Use clear, complete sentences. Avoid jargon where possible, or briefly explain it.\n"
            "3. Organize the summary with headings for main topics (e.g., '## Topic Name').\n"
            "4. Use bullet points '- ' for key details under each heading.\n"
            "5. Include ONE practical activity or thought question related to the text at the end under '## Try This'.\n"
            "6. Start the summary with the heading '# Summary'.\n"
            "7. Ensure the summary flows logically and avoids redundancy.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining a math concept to a student in grades 4-6 (ages 9-11).\n"
            "Instructions:\n"
            "1. Explain the core math concept clearly.\n"
            "2. Provide a step-by-step example of how to solve a typical problem.\n"
            "3. Briefly mention a real-world application or why it's useful.\n"
            "4. Include ONE practice problem (with answer separate or omitted) at the end under '## Practice Problem'.\n"
            "5. Start the explanation with the heading '# Math Explained'.\n"
            "6. Use headings (e.g., '## Concept', '## Example', '## Why it Matters') and bullet points '- ' for clarity.\n"
            "Text to explain:\n{text}"
        )
    },
    "higher": { # Grades 7-12
        "standard": (
            "You are creating a comprehensive summary for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Identify key themes, arguments, and supporting evidence in the text.\n"
            "2. Structure the summary logically, perhaps by theme or section, using clear headings ('## Theme/Section').\n"
            "3. Use appropriate academic vocabulary but ensure clarity.\n"
            "4. Synthesize information; do not just list points. Avoid repetition.\n"
            "5. Mention real-world implications, applications, or connections where relevant.\n"
            "6. Include ONE thought-provoking question, research idea, or analysis task related to the content at the end under '## Further Thinking'.\n"
            "7. Start the summary with the heading '# Comprehensive Summary'.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining an advanced math topic for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Provide concise definitions of key terms or concepts.\n"
            "2. Briefly outline the logic or sketch a proof if applicable.\n"
            "3. Include a non-trivial worked example demonstrating the concept or technique.\n"
            "4. Mention practical applications or connections to other fields (science, engineering, finance, etc.).\n"
            "5. Include ONE challenging problem or extension idea at the end under '## Challenge'.\n"
            "6. Start the explanation with the heading '# Advanced Math Concepts'.\n"
            "7. Use appropriate mathematical notation (like LaTeX placeholders: \\int, \\sum, etc. if present in source text) and structure with headings/subheadings.\n"
            "Text to explain:\n{text}"
        )
    }
}
# -----------------------------------------------------------------


def model_generate(prompt_text, max_new_tokens=1024, temperature=0.6):
    """Generate text using the loaded LLM."""
    # Added check for loaded model
    if not model or not tokenizer:
        st.error("LLM model not loaded. Cannot generate text.", icon="❌")
        return "Error: LLM not available."

    # Estimate prompt tokens - use a safe margin
    # Tokenizer might not have model_max_length set correctly, use a default guess if needed
    model_context_limit = getattr(tokenizer, 'model_max_length', 4096) # Default guess
    max_prompt_tokens = model_context_limit - max_new_tokens - 100 # Increased buffer

    gen_msg = f"Generating response (max_new_tokens={max_new_tokens}, temp={temperature})..."
    print(gen_msg)
    st.session_state['progress_message'] = gen_msg # Update status

    try:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens).to(device)

        if inputs['input_ids'].shape[1] >= max_prompt_tokens:
             trunc_warn = f"Warning: Prompt potentially truncated to fit model context window ({max_prompt_tokens} tokens)."
             print(trunc_warn)
             st.warning(trunc_warn, icon="⚠️")

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True if temperature > 0 else False
            )

        prompt_token_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(outputs[0][prompt_token_length:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip()

        gen_complete_msg = "...Generation complete."
        print(gen_complete_msg)
        st.session_state['progress_message'] = gen_complete_msg # Update status
        return generated_text
    except Exception as e:
        gen_err_msg = f"Error during model generation: {e}"
        print(gen_err_msg)
        traceback.print_exc()
        st.error(f"Error during summary generation: {e}", icon="❌")
        return "Error: Could not generate summary due to model issue."

# --- Use the LATEST generate_summary function provided by the user ---
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False):
    """Generate a summary meeting specific word count targets for given durations."""

    # Set exact word count ranges based on user requirements (from the user's last code version)
    if duration_minutes == 10:
        min_words = 1300
        max_words = 1500
    elif duration_minutes == 20:
        min_words = 2600
        max_words = 3000
    elif duration_minutes == 30:
        min_words = 3900
        max_words = 4000
    else:
        # Fallback or default if duration somehow invalid
        print(f"Warning: Invalid duration '{duration_minutes}'. Defaulting to 10 min targets.")
        min_words = 1300
        max_words = 1500
        # Or raise ValueError("Invalid duration_minutes. Must be 10, 20, or 30.")

    target_msg = f"Targeting summary for {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words)."
    print(target_msg)
    st.session_state['progress_message'] = target_msg

    # Combine text chunks
    full_text = ' '.join(text_chunks)
    # Estimate token count - be mindful this can be inaccurate
    full_text_tokens_estimate = len(tokenizer.encode(full_text))
    print(f"Estimated full text tokens: {full_text_tokens_estimate}")


    # --- Max Tokens Calculation (respecting user's logic) ---
    # Set max tokens based on max_words (using user's assumed 1.5 tokens per word)
    # Cap this significantly to avoid OOM, even if calculation is high
    MAX_POSSIBLE_TOKENS = 4096 # Set a hard realistic cap based on typical model limits & VRAM
    estimated_target_max_tokens = int(max_words * 1.5)
    # Use the smaller of the estimate or the hard cap, also consider model's actual max length if available
    model_context_limit = getattr(tokenizer, 'model_max_length', 4096)
    # Leave space for prompt, request AT MOST half the context limit for generation
    safe_generation_limit = min(MAX_POSSIBLE_TOKENS, model_context_limit // 2)
    max_new_tokens_summary = min(estimated_target_max_tokens, safe_generation_limit)
    max_new_tokens_summary = max(max_new_tokens_summary, 256) # Ensure a minimum generation length

    print(f"Calculated max_new_tokens for summary: {max_new_tokens_summary} (capped at {safe_generation_limit})")


    # --- Single Pass Check (respecting user's logic) ---
    # Check if estimated input tokens + estimated max output tokens fit within context window
    prompt_instruction_buffer = 600 # Increase buffer for the detailed prompts
    required_tokens_for_single_pass = full_text_tokens_estimate + max_new_tokens_summary + prompt_instruction_buffer
    # Use a safe threshold (e.g., 90% of context limit)
    can_summarize_all_at_once = required_tokens_for_single_pass < (model_context_limit * 0.9)

    # Force chunking if estimate is too high anyway, or if input text is very large
    if max_new_tokens_summary > 2048: # If asking for very long output, chunking is safer
        print("Requested output length is large (>2048 tokens), preferring chunking.")
        can_summarize_all_at_once = False
    if full_text_tokens_estimate > (model_context_limit * 0.7): # If input alone uses most of context
        print("Input text is very large, preferring chunking.")
        can_summarize_all_at_once = False


    # --- Generate initial summary (respecting user's logic) ---
    if can_summarize_all_at_once and len(text_chunks) > 0 :
        print("Attempting summary generation in a single pass.")
        st.session_state['progress_message'] = "Generating summary (single pass)..."
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        # Ensure the text replacement doesn't break if format string has issues
        try:
             prompt = prompt_template.format(text=full_text)
        except KeyError:
             st.error("Error formatting prompt template. Using basic prompt.", icon="❗")
             prompt = f"Summarize the following text for {grade_level_desc}:\n\n{full_text}"

        # Add user's specific detailed instructions for single pass
        prompt += (
            f"\n\nGenerate a detailed summary aiming for {min_words} to {max_words} words. Cover key points comprehensively."
            f" Provide in-depth explanations, examples, and elaborations on concepts."
            f" Include relevant background information and context."
        )
        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens_summary, temperature=0.7)
    elif len(text_chunks) > 0:
        chunk_start_msg = f"Text too long or requires large output. Summarizing {len(text_chunks)} chunks iteratively."
        print(chunk_start_msg)
        st.session_state['progress_message'] = chunk_start_msg

        chunk_summaries = []
        running_summary_context = ""
        # Adjust max tokens per chunk based on total target and number of chunks
        # Ensure it's capped reasonably
        max_new_tokens_chunk = min( (max_new_tokens_summary // len(text_chunks)) + 150, safe_generation_limit // 2)
        max_new_tokens_chunk = max(max_new_tokens_chunk, 150) # Min length per chunk summary
        print(f"Max tokens per chunk summary: {max_new_tokens_chunk}")

        for i, chunk in enumerate(text_chunks):
            chunk_msg = f"Summarizing chunk {i + 1}/{len(text_chunks)}..."
            print(chunk_msg)
            st.session_state['progress_message'] = chunk_msg

            prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
            context = f"\n\nContext from previous summary parts:\n{running_summary_context[-600:]}" if running_summary_context else "" # Slightly more context

            try:
                 chunk_prompt = prompt_template.format(text=chunk) + context
            except KeyError:
                 st.error(f"Error formatting prompt template for chunk {i+1}. Using basic prompt.", icon="❗")
                 chunk_prompt = f"Summarize this part of the text for {grade_level_desc}:\n\n{chunk}\n\n{context}"

            # Add user's specific detailed instructions for chunk pass
            chunk_prompt += (
                f"\n\nSummarize this part ({max_new_tokens_chunk} tokens max). It is piece {i + 1} of {len(text_chunks)}. "
                f"The final summary should be detailed ({min_words}-{max_words} words). "
                f"Include specific examples and explain the significance of points in this chunk."
            )

            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_new_tokens_chunk, temperature=0.5)
            # Clean chunk summary more carefully
            chunk_summary = re.sub(r'^.*?#+\s*.*?\s*#*', '', chunk_summary) # Remove potential headings/instructions
            chunk_summary = '. '.join(s.strip() for s in chunk_summary.split('.') if len(s.strip()) > 15) # Keep longer fragments
            if chunk_summary:
                if not chunk_summary.endswith('.'): chunk_summary += '.'
                chunk_summaries.append(chunk_summary)
                running_summary_context += chunk_summary + "\n"
            else:
                 print(f"  Warning: Chunk {i+1} summary was empty/short after cleaning.")


        if not chunk_summaries:
             st.error("Failed to generate summaries from any text chunks.", icon="❌")
             return "Error: No valid chunk summaries were generated."

        # Consolidate chunks
        consol_msg = "Consolidating chunk summaries..."
        print(consol_msg)
        st.session_state['progress_message'] = consol_msg
        # Calculate max tokens for consolidation based on overall target, capped
        max_tokens_consolidation = min(int(max_words * 1.6), safe_generation_limit)
        max_tokens_consolidation = max(max_tokens_consolidation, 512) # Min consolidation length

        consolidation_prompt = (
            f"Consolidate these summary pieces into one coherent, detailed final summary for {grade_level_desc}.\n"
            f"Ensure the summary flows logically, removes redundancy, and is approximately {min_words} to {max_words} words long.\n"
            f"Structure with appropriate headings (e.g., starting with '# Comprehensive Summary') and bullet points where suitable.\n"
            f"Provide extensive coverage, rich details, and explanations. Include a final activity/question section.\n\n"
            f"Here are the summary pieces (separated by '---'):\n\n" + "\n\n---\n\n".join(chunk_summaries) +
            f"\n\n--- End of Pieces ---\n\n"
            f"Create the final, unified, detailed summary (approx. {min_words}-{max_words} words, {max_tokens_consolidation} tokens max):"
        )
        initial_summary = model_generate(consolidation_prompt, max_new_tokens=max_tokens_consolidation, temperature=0.7)
    else:
        st.error("No text chunks found to summarize.", icon="❌")
        return "Error: Text processing resulted in zero chunks."


    # --- Iterative lengthening (respecting user's logic) ---
    current_summary = initial_summary
    # Clean up potential error messages before checking length
    if current_summary.startswith("Error:"):
        st.error(f"Initial summary generation failed: {current_summary}", icon="❌")
        return current_summary

    continuation_prompt_template = (
        "This summary is currently too short. Continue elaborating on the existing points to make it more detailed and comprehensive.\n"
        "Provide more specific examples, explain the reasoning behind concepts, add relevant background or related information, and expand on the implications.\n"
        "Maintain coherence with the preceding text.\n"
        "Summary Text (End Portion):\n...{last_part}\n\n"
        "Detailed Continuation:"
    )

    attempts = 0
    max_lengthening_attempts = 5 # Limit attempts
    while len(current_summary.split()) < min_words and attempts < max_lengthening_attempts:
        current_word_count = len(current_summary.split())
        lengthen_msg = f"Summary at {current_word_count} words; extending towards {min_words} (Attempt {attempts + 1}/{max_lengthening_attempts})."
        print(lengthen_msg)
        st.session_state['progress_message'] = lengthen_msg

        # Take a reasonably sized snippet from the end for context
        context_words = 300
        last_part = ' '.join(current_summary.split()[-context_words:])

        # Request a moderate amount of new tokens per iteration
        tokens_to_add = min(500, max_new_tokens_summary) # Don't ask for more than initial max_tokens
        prompt = continuation_prompt_template.format(last_part=last_part)

        new_part = model_generate(prompt, max_new_tokens=tokens_to_add, temperature=0.7)

        if new_part.startswith("Error:"):
            st.error("Failed to generate continuation.", icon="❌")
            break # Stop trying if generation fails

        # Basic cleaning of the new part
        new_part = re.sub(r'^.*?Detailed Continuation:', '', new_part, flags=re.IGNORECASE).strip()
        if new_part and len(new_part.split()) > 5: # Only add if it's substantial
            current_summary += " " + new_part
            print(f"  Added {len(new_part.split())} words.")
        else:
            print("  Continuation generation yielded little/no new text. Stopping lengthening.")
            break # Stop if LLM isn't adding anything useful
        attempts += 1

    if attempts == max_lengthening_attempts and len(current_summary.split()) < min_words:
        st.warning(f"Reached max attempts ({max_lengthening_attempts}) but summary word count ({len(current_summary.split())}) is still below target ({min_words}).", icon="⚠️")


    # --- Trim if necessary (respecting user's logic) ---
    words = current_summary.split()
    final_word_count_before_trim = len(words)
    if final_word_count_before_trim > max_words:
        trim_msg = f"Summary at {final_word_count_before_trim} words; trimming to target maximum of {max_words} words."
        print(trim_msg)
        st.session_state['progress_message'] = trim_msg
        # Simple trim from the end
        current_summary = ' '.join(words[:max_words])
        # Add ellipsis to indicate truncation
        if not current_summary.endswith('.'): current_summary += "..."
        else: current_summary = current_summary[:-1] + "..."


    summary = current_summary

    # --- Post-process ---
    post_process_msg = "Post-processing final summary..."
    print(post_process_msg)
    st.session_state['progress_message'] = post_process_msg
    processed_summary = enhanced_post_process(summary, grade_level_category, min_words, max_words)

    # --- Ensure activity section ---
    # Use regex with MULTILINE flag for headings
    if not re.search(r'^##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity)', processed_summary, re.IGNORECASE | re.MULTILINE):
        activity_msg = "Adding missing activity/question section..."
        print(activity_msg)
        st.session_state['progress_message'] = activity_msg
        activity = generate_activity(processed_summary, grade_level_category, grade_level_desc)
        activity_heading_map = {
            "lower": "## Fun Activity",
            "middle": "## Try This",
            "higher": "## Further Thinking"
        }
        activity_heading = activity_heading_map.get(grade_level_category, "## Activity Suggestion")
        processed_summary += f"\n\n{activity_heading}\n{activity}"

    final_word_count = len(processed_summary.split())
    final_msg = f"Final summary generated ({final_word_count} words)."
    print(final_msg)
    st.session_state['progress_message'] = final_msg
    return processed_summary

# --- Use the LATEST enhanced_post_process function provided by the user ---
def enhanced_post_process(summary, grade_level_category, min_words, max_words):
    """Advanced post-processing including structure, completion, and deduplication."""

    # Ensure consistent main heading (using expected prompts)
    main_heading_text = prompts[grade_level_category]["standard"].splitlines()[0].split("#")[-1].strip()
    summary = re.sub(
        r'^#+\s*(Summary|Simple Summary|Math Fun|Math Explained|Comprehensive Summary|Advanced Math Concepts)\s*#*',
        f'# {main_heading_text}',
        summary, count=1, flags=re.IGNORECASE | re.MULTILINE).strip()
    if not summary.startswith("# "):
        summary = f'# {main_heading_text}\n\n' + summary

    lines = summary.split('\n')
    processed_lines = []
    seen_content = set() # Basic check for exact duplicates
    elaboration_candidates = [] # Points that might benefit from user's elaboration logic

    # First pass: clean lines, handle basic structure, identify candidates for elaboration
    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith('#'): # Keep headings
            processed_lines.append(line)
            continue

        is_bullet = line.startswith('- ')
        content = line[2:] if is_bullet else line
        content = content.strip()
        if not content: continue

        # Simple exact duplicate check
        content_key = content[:50].lower() # Key based on start of line
        if content_key in seen_content:
            print(f"  Skipping duplicate line: {content[:50]}...")
            continue
        seen_content.add(content_key)

        # Attempt sentence completion (only if clearly a fragment)
        is_fragment = not re.search(r'[.!?]$', content) and len(content.split()) > 3 and content[0].isupper()
        if is_fragment:
            # print(f"Attempting to complete fragment: '{content}'") # Can be noisy
            content = complete_sentence(content) # complete_sentence adds punctuation

        # Ensure proper sentence casing
        if content and content[0].islower():
            content = content[0].upper() + content[1:]

        # Ensure sentence ends with punctuation (if not already done by completion)
        if content and content[-1].isalnum():
             content += '.'

        # Store line/content for next pass
        processed_lines.append({"text": content, "is_bullet": is_bullet})

        # Identify candidates for user's elaboration logic (short, non-bullet points)
        if len(content.split()) < 15 and not is_bullet:
            elaboration_candidates.append(content)


    # User's elaboration logic (optional call to model)
    # Limit calls to model here to avoid excessive processing time
    if elaboration_candidates and len(processed_lines) < 30: # Only elaborate if summary isn't already very long
        elaboration_msg = f"Attempting elaboration on {len(elaboration_candidates)} short points..."
        print(elaboration_msg)
        st.session_state['progress_message'] = elaboration_msg

        elaboration_prompt = (
            "Please elaborate on the following points extracted from a summary. Provide more details, examples, or explanations for each.\n"
            "Keep the elaborations concise and relevant to the likely context.\n"
            "Format each elaboration as a separate sentence or two.\n\n"
            "Points to Elaborate On:\n" + "\n".join([f"- {p}" for p in elaboration_candidates]) + "\n\nElaborations:"
        )
        # Limit max_new_tokens for elaboration to avoid runaways
        elaborations_text = model_generate(elaboration_prompt, max_new_tokens=min(len(elaboration_candidates) * 50, 400), temperature=0.6)

        if not elaborations_text.startswith("Error:"):
             # Add elaborations as new bullet points
             elaboration_lines = [e.strip() for e in elaborations_text.split('\n') if len(e.strip()) > 10]
             print(f"  Generated {len(elaboration_lines)} elaboration lines.")
             for elab_line in elaboration_lines:
                 # Ensure punctuation
                 if elab_line[-1].isalnum(): elab_line += '.'
                 processed_lines.append({"text": elab_line, "is_bullet": True}) # Add as bullet points
        else:
             print("  Elaboration generation failed.")


    # Semantic Deduplication and final assembly
    final_processed_lines = []
    points_for_dedup = []
    line_indices_for_dedup = {} # Map content back to original processed_lines index

    for i, line_data in enumerate(processed_lines):
         # Collect points suitable for semantic deduplication (ignore headings)
         if not line_data["text"].startswith('#'):
              # Use sentences within the line text for finer-grained deduplication
              sentences = re.split(r'(?<=[.!?])\s+', line_data["text"])
              for sent in sentences:
                   sent = sent.strip()
                   if len(sent.split()) > 5: # Only consider reasonably long sentences
                        points_for_dedup.append(sent)
                        # Store which original processed line this sentence came from
                        if sent not in line_indices_for_dedup:
                             line_indices_for_dedup[sent] = []
                        line_indices_for_dedup[sent].append(i)


    unique_points_content = points_for_dedup # Default if dedup fails or skipped
    if points_for_dedup:
        dedup_msg = f"Running semantic duplicate removal on {len(points_for_dedup)} potential points/sentences..."
        print(dedup_msg)
        st.session_state['progress_message'] = dedup_msg
        unique_points_content = remove_duplicates_semantic(points_for_dedup)
        print(f"Reduced to {len(unique_points_content)} unique points/sentences.")
    else:
        print("Skipping semantic duplicate removal (no suitable points found).")

    # Rebuild the summary using unique points, preserving original structure (headings, bullets)
    kept_line_indices = set()
    unique_content_set = set(unique_points_content)

    # Keep headings first
    for i, line_data in enumerate(processed_lines):
         if line_data["text"].startswith('#'):
              final_processed_lines.append(line_data["text"])
              kept_line_indices.add(i)

    # Keep lines whose sentences were deemed unique
    for sent, indices in line_indices_for_dedup.items():
        if sent in unique_content_set:
            for index in indices:
                 if index not in kept_line_indices: # Avoid adding the same line multiple times
                     line_data = processed_lines[index]
                     formatted_line = f"- {line_data['text']}" if line_data['is_bullet'] else line_data['text']
                     final_processed_lines.append(formatted_line)
                     kept_line_indices.add(index)

    # Assemble the final text
    full_text = ""
    for i, line in enumerate(final_processed_lines):
        full_text += line
        # Add spacing - double newline after headings, single otherwise
        is_heading = line.startswith('#')
        if is_heading and i < len(final_processed_lines) - 1:
            full_text += "\n\n"
        elif i < len(final_processed_lines) - 1:
             # Add extra newline if next line is also not a heading (separates paragraphs/bullets)
             is_next_heading = (i + 1 < len(final_processed_lines)) and final_processed_lines[i+1].startswith('#')
             if not is_heading and not is_next_heading:
                  full_text += "\n" # Single newline between bullets/paragraphs
             else:
                  full_text += "\n" # Single newline before heading or at end

    return full_text.strip()


def remove_duplicates_semantic(points, similarity_threshold=0.90):
    """Remove near-duplicate points using sentence embeddings."""
    if not points or not embedder: # Check if embedder loaded
        print("Warning: Embedder not available or no points; skipping semantic dedup.")
        return points
    try:
        # Filter out very short points before encoding
        valid_points = [p for p in points if len(p.split()) > 3]
        if not valid_points: return []

        embeddings = embedder.encode(valid_points, convert_to_tensor=True, show_progress_bar=False)
        unique_indices = []
        kept_embeddings = [] # Store embeddings of points we decide to keep

        for i in range(len(valid_points)):
            is_duplicate = False
            if kept_embeddings:
                # Calculate similarity with all previously kept points efficiently
                all_kept_embeddings = torch.cat(kept_embeddings, dim=0)
                # Use util.dot_score for potentially faster calculation on normalized embeddings (MiniLM is normalized)
                similarities = util.dot_score(embeddings[i], all_kept_embeddings)[0] # Get scores for current item vs all kept
                # Check if the maximum similarity found exceeds the threshold
                if torch.max(similarities) > similarity_threshold:
                    is_duplicate = True
                    # print(f"    Dedup: '{valid_points[i][:50]}...' is similar to a kept point (max sim: {torch.max(similarities):.3f})")


            if not is_duplicate:
                unique_indices.append(i)
                kept_embeddings.append(embeddings[i].unsqueeze(0)) # Add the new unique embedding

        return [valid_points[i] for i in unique_indices]
    except Exception as e:
        print(f"Error during semantic duplicate removal: {e}. Returning original points.")
        traceback.print_exc() # Log detailed error
        return points


def generate_activity(summary_text, grade_level_category, grade_level_desc):
    """Dynamically generate a relevant activity using the LLM."""
    # Added check for loaded model
    if not model or not tokenizer:
        print("Warning: LLM not loaded, cannot generate activity.")
        return "- Discuss the main idea with someone." # Basic fallback

    activity_msg = "Generating relevant activity..."
    print(activity_msg)
    st.session_state['progress_message'] = activity_msg

    activity_prompt_template = (
        "Based on the preceding summary content, suggest ONE simple and engaging activity suitable for {grade_desc}.\n"
        "The activity must be directly related to the core topics of the summary.\n"
        "Describe the activity clearly in one or two actionable sentences.\n\n"
        "Summary Context (End Portion):\n...{summary}\n\n"
        "Activity Suggestion:"
    )
    summary_snippet = summary_text[-1000:] # Use last 1000 chars as more relevant context
    prompt = activity_prompt_template.format(grade_desc=grade_level_desc, summary=summary_snippet)

    activity = model_generate(prompt, max_new_tokens=100, temperature=0.7)

    # Cleaning
    activity = activity.strip().replace("Activity Suggestion:", "").strip()
    # Remove potential leading bullet if model adds one
    activity = re.sub(r'^[\-\*\s]+', '', activity)

    if activity and not activity.startswith("Error:"):
        activity = activity[0].upper() + activity[1:]
        # Add bullet point formatting
        activity = f"- {activity}"
        if activity[-1].isalnum(): activity += '.'
        return activity
    else:
        print("Warning: Failed to generate activity or generation error.")
        # Generic fallback
        fallbacks = { "lower": "- Draw a picture about what you learned!",
                      "middle": "- Try to explain the main idea to a friend.",
                      "higher": "- Think of one question you still have about this topic." }
        return fallbacks.get(grade_level_category, "- Review the key points.")


# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("📚 Enhanced PDF Summarizer")
st.markdown("Upload a PDF, select the target grade level and desired summary duration, and get a tailored summary generated by Llama 3.")

# Initialize session state for progress message
if 'progress_message' not in st.session_state:
    st.session_state['progress_message'] = "Ready"
if 'summary_output' not in st.session_state:
    st.session_state['summary_output'] = ""


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("1. Upload PDF Document", type="pdf")

    grade = st.slider("2. Target Grade Level", min_value=1, max_value=12, value=6, help="Select the approximate grade level of the intended reader.")

    duration_options = {
        "Approx. 10 Minutes (~1300-1500 words)": 10,
        "Approx. 20 Minutes (~2600-3000 words)": 20,
        "Approx. 30 Minutes (~3900-4000 words)": 30
    }
    duration_label = st.radio(
        "3. Target Summary Length (Reading Time)",
        options=list(duration_options.keys()),
        index=1, # Default to 20 minutes
        help="Select the desired length of the summary based on approximate reading time and word count goals."
    )
    duration = duration_options[duration_label]

    # Advanced options in expander
    with st.expander("Advanced Options"):
        # Use a global variable controlled by this checkbox
        # The actual modification happens within extract_text_from_pdf using 'global'
        ocr_enabled_checkbox = st.checkbox("Enable Image OCR", value=False, help="Extract text from images within the PDF (slower, requires Tesseract).")
        chunk_size = st.number_input("Chunk Size (words)", min_value=100, max_value=2000, value=500, step=50, help="Approximate words per text chunk for processing long documents.")
        overlap = st.number_input("Chunk Overlap (words)", min_value=0, max_value=500, value=50, step=10, help="Word overlap between chunks.")

    # Generate Button
    generate_button = st.button("✨ Generate Summary ✨", type="primary", use_container_width=True)

    # Status Area in Sidebar
    st.divider()
    st.subheader("Status")
    status_placeholder = st.empty()
    status_placeholder.info(st.session_state['progress_message'])


# --- Main Area for Output ---
output_placeholder = st.empty()

if generate_button:
    if uploaded_file is not None:
        # Set OCR global flag based on checkbox BEFORE calling processing
        USE_IMAGE_OCR = ocr_enabled_checkbox
        st.session_state['summary_output'] = "" # Clear previous output

        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        try:
            st.session_state['progress_message'] = "Processing PDF..."
            status_placeholder.info(st.session_state['progress_message']) # Update status immediately

            grade_level_category, grade_level_desc = determine_reading_level(grade)
            st.session_state['progress_message'] = f"Targeting: {grade_level_desc}"
            status_placeholder.info(st.session_state['progress_message'])

            # --- Core Processing Steps ---
            raw_text = extract_text_from_pdf(pdf_path, detect_math=True) # Status updated inside function

            if not raw_text or len(raw_text.strip()) < 50:
                st.error("No significant text could be extracted from the PDF.", icon="🚫")
            else:
                has_math = detect_math_content(raw_text)
                st.session_state['progress_message'] = f"Math content detected: {has_math}"
                status_placeholder.info(st.session_state['progress_message'])

                st.session_state['progress_message'] = "Cleaning extracted text..."
                status_placeholder.info(st.session_state['progress_message'])
                cleaned_text = clean_text(raw_text)

                st.session_state['progress_message'] = "Splitting text into chunks..."
                status_placeholder.info(st.session_state['progress_message'])
                # Use chunk_size and overlap from widgets
                chunks = split_text_into_chunks(cleaned_text, chunk_size=chunk_size, overlap=overlap)

                if not chunks:
                    st.error("Failed to split text into processable chunks.", icon="🚫")
                else:
                    st.session_state['progress_message'] = "Starting summary generation..."
                    status_placeholder.info(st.session_state['progress_message'])

                    summary = generate_summary(
                        chunks,
                        grade_level_category,
                        grade_level_desc,
                        duration, # Duration value (10, 20, 30)
                        has_math
                    ) # Status updated inside function

                    st.session_state['summary_output'] = summary # Store summary in session state

                    # Check if summary generation failed
                    if summary.startswith("Error:"):
                         st.error(f"Summary generation failed: {summary}", icon="❌")
                         st.session_state['progress_message'] = "Failed."
                         status_placeholder.error(st.session_state['progress_message'])

                    else:
                         final_word_count = len(summary.split())
                         success_msg = f"✅ Summary Generation Complete ({final_word_count} words)."
                         st.success(success_msg, icon="🎉")
                         st.session_state['progress_message'] = "Complete."
                         status_placeholder.success(st.session_state['progress_message'])


        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}", icon="🔥")
            traceback.print_exc() # Log detailed traceback to console
            st.session_state['progress_message'] = "Error occurred."
            status_placeholder.error(st.session_state['progress_message'])
        finally:
            # Clean up temporary file
            if 'pdf_path' in locals() and os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"Temporary file removed: {pdf_path}")

    else:
        st.warning("Please upload a PDF file first.", icon="⚠️")
        st.session_state['progress_message'] = "Waiting for PDF."
        status_placeholder.warning(st.session_state['progress_message'])


# --- Display Summary Output (if available in session state) ---
if st.session_state['summary_output'] and not st.session_state['summary_output'].startswith("Error:"):
     with output_placeholder.container(): # Use container to redraw output area
        st.subheader("Generated Summary")
        st.markdown(f"**Target Audience:** {determine_reading_level(grade)[1]} | **Target Length:** {duration} minutes")
        st.text_area("Summary Text", st.session_state['summary_output'], height=500)

        # Generate default filename for download
        download_filename = "summary.txt"
        if uploaded_file:
            base_name = os.path.splitext(uploaded_file.name)[0]
            download_filename = f"{base_name}_grade{grade}_summary.txt"

        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state['summary_output'].encode('utf-8'), # Encode to bytes
            file_name=download_filename,
            mime="text/plain"
        )
else:
    # Display a message if no summary is generated yet or if there was an error previously handled
     with output_placeholder.container():
        if not generate_button: # Only show initial message if button hasn't been pressed
             st.info("Upload a PDF and click 'Generate Summary' to begin.")


# --- End of File streamlit_app.py ---