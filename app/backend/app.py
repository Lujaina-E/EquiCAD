from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import redis
import json
import io
import re
from ingestion.table_extraction import extract_tables_with_fallback
import fitz  
import PyPDF2
from datetime import datetime, timezone
import uuid
from flask import send_from_directory
import nltk
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

CORS(
    app,
    origins=[
        "https://equicad-production.up.railway.app",
        "https://netlify-equicad-frontend.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    supports_credentials=True
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment")

_client = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        _client = OpenAI(api_key=api_key)
    return _client


# Configuration
FINE_TUNED_MODEL_ID = os.getenv('FINE_TUNED_MODEL_ID')
OCR_MODEL_ID = os.getenv('OCR_MODEL_ID', 'gpt-4o')
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TEXT_WORDS = 200


REDIS_URL = os.environ.get("REDIS_URL")

if not REDIS_URL:
    raise RuntimeError("REDIS_URL environment variable is not set")

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True
)

try:
    redis_client.ping()
except redis.exceptions.RedisError as e:
    raise RuntimeError(f"Redis connection failed: {e}")

SESSION_TTL_SECONDS = 60 * 60  # 1 hour
FILE_TTL_SECONDS = 60 * 60  


def get_session(session_id):
    raw = redis_client.get(f"session:{session_id}")
    return json.loads(raw) if raw else None


def save_session(session_id, state):
    redis_client.setex(
        f"session:{session_id}",
        SESSION_TTL_SECONDS,
        json.dumps(state)
    )


def delete_session(session_id):
    redis_client.delete(f"session:{session_id}")


def get_file(file_id):
    raw = redis_client.get(f"file:{file_id}")
    return json.loads(raw) if raw else None


def save_file(file_id, file_data):
    redis_client.setex(
        f"file:{file_id}",
        FILE_TTL_SECONDS,
        json.dumps(file_data)
    )


def delete_file(file_id):
    redis_client.delete(f"file:{file_id}")


def extract_pages_from_pdf(file_stream):
    """Extract text from each page of PDF"""
    pdf_reader = PyPDF2.PdfReader(file_stream)
    pages = []

    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "page_num": page_num,
            "text": text.strip()
        })

    return pages


def extract_caption(text):
    """Extract caption from text"""
    if not text:
        return "Caption not detected"
    lines = text.splitlines()
    for line in lines[:5]:
        if re.search(r"(table|figure)\s*\d+", line, re.I):
            return line.strip()
    return "Caption not detected"


def extract_all_figure_candidates(file_stream):
    """Extract all potential figures from PDF"""
    file_stream.seek(0)
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    file_stream.seek(0)
    figures = []

    # --- Preload embedded files at document level ---
    embedded_files = []

    if hasattr(doc, "embeddedFileNames"):  # PyMuPDF 1.x
        for fname in doc.embeddedFileNames():
            file_dict = doc.embeddedFileGet(fname)
            embedded_files.append({
                "filename": fname,
                "data": file_dict["data"]
            })

    elif hasattr(doc, "embedded_files"):  # PyMuPDF 2.x
        for fname, file_dict in doc.embedded_files().items():
            embedded_files.append({
                "filename": fname,
                "data": file_dict["data"]
            })
            
    for page_num, page in enumerate(doc, start=1):
        # 1. Raster images 
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                figures.append({
                    "page": page_num,
                    "type": "image",
                    "data": image_bytes
                })
            except Exception as e:
                print(f"Failed to extract image on page {page_num}: {e}")

        # 2. Vector graphics 
        try:
            vector_blocks = page.get_drawings()
            if vector_blocks:
                figures.append({
                    "page": page_num,
                    "type": "vector_graphic",
                    "data": vector_blocks
                })
        except Exception as e:
            print(f"Failed to extract vector graphics on page {page_num}: {e}")

        # 3. Embedded PDFs 
        if page_num == 1:
            for emb in embedded_files:
                figures.append({
                    "page": page_num,
                    "type": "embedded_pdf",
                    "filename": emb["filename"],
                    "data": emb["data"]
                })

        # 4. Text-based captions 
        text = page.get_text("text")
        figure_text_lines = [
            line for line in text.splitlines()
            if re.search(r"figure\s*\d+", line, re.I)
        ]
        if figure_text_lines:
            figures.append({
                "page": page_num,
                "type": "caption_text",
                "data": "\n".join(figure_text_lines)
            })

    return figures


def perform_ocr_on_image(image_bytes):
    """Temporary OCR stub"""
    return "<OCR not implemented>"


def split_sentences(text):
    """Split text into sentences using NLTK."""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def split_paragraphs(text):
    lines = [l.strip() for l in text.replace('\r\n', '\n').split('\n') if l.strip()]
    paragraphs = []
    current_para = []

    for i, line in enumerate(lines):
        current_para.append(line)

        # Lookahead to next line
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            # Heuristic: next line starts with capital letter, current line ends with punctuation
            if (line.endswith(('.', '!', '?')) and next_line and next_line[0].isupper()) or len(line) < 40 and next_line[0].isupper():
                paragraphs.append(' '.join(current_para))
                current_para = []

    # Add last paragraph
    if current_para:
        paragraphs.append(' '.join(current_para))

    return [p for p in paragraphs if p]


def split_sections(text):
    """Split text into sections based on headers."""
    lines = text.split('\n')
    sections = []
    current_section = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_all_caps = stripped.isupper() and len(stripped.split()) > 1
        is_numbered = bool(re.match(r'^\d+(\.\d+)*\s', stripped))
        ends_with_colon = stripped.endswith(':')

        if is_all_caps or is_numbered or ends_with_colon:
            if current_section:
                sections.append('\n'.join(current_section).strip())
                current_section = []

        current_section.append(stripped)

    if current_section:
        sections.append('\n'.join(current_section).strip())

    return sections if sections else [text]

def split_text_by_granularity(text, granularity='sectional'):
    """Split text into chunks based on granularity."""
    text = text.strip()
    if not text:
        return []

    if granularity == 'sentence':
        # split each paragraph into sentences
        paragraphs = split_paragraphs(text)
        sentences = []
        for para in paragraphs:
            sentences.extend(sent_tokenize(para))
        return sentences

    elif granularity == 'paragraph':
        # split by paragraphs (sections already preserved by previous step)
        return split_paragraphs(text)

    elif granularity == 'sectional':
        return split_sections(text)

    else:
        return [text]

def normalize_file_content(file_data, granularity='sectional'):
    """
    Convert all text, tables, figures into a uniform list of chunks.
    Each chunk has: type, content, caption (optional), page (optional)
    """
    content_to_analyze = []

    # --- Text ---
    text_chunks = split_text_by_granularity(file_data.get('text_content', ''), granularity)
    for t in text_chunks:
        content_to_analyze.append({
            "type": "text",
            "content": t,
            "caption": None,
            "page": None
        })

    # --- Tables ---
    for t in file_data.get('tables_jsonl', []):
        t_obj = t if isinstance(t, dict) else json.loads(t)
        content_to_analyze.append({
            "type": "table",
            "content": t_obj.get("content", {}),
            "caption": t_obj.get("content", {}).get("caption", "No caption"),
            "page": t_obj.get("source", {}).get("page", None)
        })

    # --- Figures ---
    for f in file_data.get('figures_jsonl', []):
        f_obj = f if isinstance(f, dict) else json.loads(f)
        content_to_analyze.append({
            "type": "figure",
            "content": f_obj.get("content", {}),
            "caption": f_obj.get("content", {}).get("caption", "No caption"),
            "page": f_obj.get("source", {}).get("page", None)
        })

    return content_to_analyze


SYSTEM_MESSAGE = """You are an inference-time assistant for detecting sex-based discrimination against women in Coronary Artery Disease (CAD).

Your task:
- Analyze the input.
- ONLY assign a label and category IF the input meaningfully relates to sex-based bias or analysis in Coronary Artery Disease (CAD).
- Do NOT force a label or category if the input is unrelated to CAD.

If the input IS related to CAD, respond in EXACTLY this format (no extra text):

Label: [Bias OR No Bias]
Category: [one category from the list below]

If Label is "Bias", Category must be ONE of:
- Sampling Bias
- Diagnostic Uncertainty / Bias
- Symptom Misinterpretation

If Label is "No Bias", Category must be ONE of:
- Biological / Physiological Differences
- Factual / Neutral Observed Outcome

If the input is NOT related to Coronary Artery Disease in any meaningful way, respond with EXACTLY this single line:

No label assigned! Your input is not related to Coronary Artery Disease.

CRITICAL RULES:
1. Use EXACT wording and capitalization
2. Output EITHER the two-line Label/Category format OR the single no-label sentence — nothing else
3. Do NOT explain your reasoning
"""


def wrap_input_for_model(content_obj):
    """Wrap content into model prompt"""
    if isinstance(content_obj, str):
        user_content = f"Text: {content_obj.strip()}"
    else:
        user_content = f"{content_obj.get('type', 'content').capitalize()}: {json.dumps(content_obj, ensure_ascii=False)}"

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content}
    ]
    return messages


def extract_label_category(text: str):
    """Extract Label and Category from assistant content"""
    label_match = re.search(
        r"(?:Assigned Label|Label):\s*\[?(Bias|No Bias)\]?",
        text,
        re.IGNORECASE
    )

    category_match = re.search(
        r"Category:\s*\[?([^\]\n\r]+)\]?",
        text,
        re.IGNORECASE
    )

    if not label_match:
        raise ValueError(f"Cannot extract label from this section. Try again later.")
    if not category_match:
        raise ValueError(f"Cannot extract category from: {text}")

    label = label_match.group(1).strip()
    category = category_match.group(1).strip()
    return label, category


def detect_bias_with_model(input_obj, output_format="label_category"):
    """Send input to the fine-tuned model and parse the output"""
    messages = wrap_input_for_model(input_obj)

    try:
        response = get_openai_client().chat.completions.create(
            model=FINE_TUNED_MODEL_ID,
            messages=messages,
            temperature=0,
            max_tokens=200
        )

        assistant_text = response.choices[0].message.content.strip()
        
        if assistant_text.lower().startswith("no label assigned!"):
            return "\nNo label assigned! Your input is not related to Coronary Artery Disease."

        label, category = extract_label_category(assistant_text)

    except Exception as e:
        print("🔥 MODEL FAILURE:", str(e))
        raise

    if output_format == "label":
        return f"\nLabel: {label}"
    else:
        return f"\nLabel: {label}\nCategory: {category}"

@app.route('/api/chat/start', methods=['POST'])
def start_conversation():
    """Initialize a new conversation session"""
    session_id = str(uuid.uuid4())
    
    state = {
        'state': 'initial',
        'file_id': None,
        'parsed_content': None,
        'content_type': None,
        'granularity': None,
        'output_format': None
    }
    save_session(session_id, state)
    
    return jsonify({
        "success": True,
        "session_id": session_id,
        "message": "Welcome! How would you like to proceed?",
        "options": [
            {"id": "upload_file", "text": "Upload File"},
            {"id": "single_text", "text": "Analyze Text"}
        ]
    })


@app.route('/api/chat/upload', methods=['POST'])
def upload_file():
    """Handle file upload with robust table/figure extraction"""
    try:
        session_id = request.form.get('session_id')
        state = get_session(session_id)
        if not state:
            return jsonify({
                "success": False,
                "error": "Session expired or invalid. Please start a new analysis."
            }), 400

        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "error": "Only PDF files are supported"}), 400

        # Check file size
        file.seek(0, os.SEEK_END)
        if file.tell() > MAX_FILE_SIZE:
            return jsonify({"success": False, "error": "File too large (max 10MB)"}), 400
        file.seek(0)

        # Generate file ID and read bytes
        file_id = str(uuid.uuid4())
        file_bytes = file.read()

        # Extract pages
        pages = extract_pages_from_pdf(io.BytesIO(file_bytes))
        full_text = "\n".join(p["text"] for p in pages)

        if not full_text.strip():
            return jsonify({
                "success": False,
                "error": "The PDF appears to be empty or unreadable. Please upload a valid CAD-related document."
            }), 400

        # --- Robust Table Extraction ---
        try:
            all_table_jsonl = extract_tables_with_fallback(file_bytes, pages, file_id)
        except Exception as e:
            print(f"Table extraction error: {e}")
            all_table_jsonl = []

        # --- Figure Extraction ---
        try:
            figure_candidates = extract_all_figure_candidates(io.BytesIO(file_bytes))
        except Exception as e:
            print("DEBUG: Figure extraction failed:", str(e))
            figure_candidates = []

        all_figure_jsonl = []
        for candidate in figure_candidates:
            try:
                ocr_text = None
                if candidate["type"] == "image":
                    ocr_text = perform_ocr_on_image(candidate["data"])
                elif candidate["type"] in ["vector_graphic", "embedded_pdf"]:
                    ocr_text = "<vector/embedded figure - OCR not implemented>"
                elif candidate["type"] == "caption_text":
                    ocr_text = candidate["data"]

                figure_content = {
                    "caption": extract_caption(ocr_text) if ocr_text else "No caption",
                    "figure": {
                        "figure_type": candidate["type"],
                        "overall_description": ocr_text or "No description",
                        "axes": None,
                        "legend": None,
                        "panels": None
                    }
                }

                figure_jsonl = json.dumps({
                    "type": "figure",
                    "content": figure_content,
                    "source": {
                        "page": candidate["page"],
                        "file_id": file_id
                    }
                })

                all_figure_jsonl.append(figure_jsonl)
            except Exception as e:
                print("DEBUG: Skipping figure due to error:", str(e))

        # Save file to Redis
        file_data = {
            'filename': file.filename,
            'pages': pages,
            'text_content': full_text,
            'tables_jsonl': all_table_jsonl,
            'figures_jsonl': all_figure_jsonl,
            'upload_time': datetime.now(timezone.utc).isoformat(),
        }
        save_file(file_id, file_data)

        preview = full_text[:500] + "..." if len(full_text) > 500 else full_text

        # Update session state
        state['file_id'] = file_id
        state['state'] = 'file_uploaded'
        save_session(session_id, state)

        return jsonify({
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully. Here's a preview of the extracted content:",
            "preview": preview,
            "stats": {
                "total_tables": len(all_table_jsonl),
                "total_figures": len(all_figure_jsonl),
                "total_pages": len(pages)
            },
            "options": [
                {"id": "all", "text": "Analyze All Content"},
                {"id": "text", "text": "Analyze Text Only"},
                {"id": "table", "text": "Analyze Tables"},
                {"id": "figure", "text": "Analyze Figures"}
            ],
            "question": "What content would you like to analyze?"
        })

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chat/select-content', methods=['POST'])
def select_content_type():
    """Handle content type selection"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        content_type = data.get('content_type')
        
        state = get_session(session_id)
        if not state:
            return jsonify({
                "success": False,
                "error": "Session expired or invalid. Please start a new analysis."
            }), 400
        
        state['content_type'] = content_type
        
        if content_type == 'text':
            state['state'] = 'awaiting_granularity'
            save_session(session_id, state)
            return jsonify({
                "success": True,
                "message": "You selected text analysis. What granularity would you like?",
                "options": [
                    {"id": "sectional", "text": "Sectional Level"},
                    {"id": "paragraph", "text": "Paragraph Level"},
                    {"id": "sentence", "text": "Sentence Level"}
                ]
            })
        
        # For table/figure/all, skip granularity and go to output format
        state['granularity'] = 'sectional'  # default
        state['state'] = 'awaiting_output_format'
        save_session(session_id, state)

        return jsonify({
            "success": True,
            "message": "How would you like the results displayed?",
            "options": [
                {"id": "label", "text": "Label Only"},
                {"id": "label_category", "text": "Label + Category"}
            ]
        })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chat/select-granularity', methods=['POST'])
def select_granularity():
    """Handle granularity selection for text analysis"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        granularity = data.get('granularity')
        
        state = get_session(session_id)
        if not state:
            return jsonify({"success": False, "error": "Session expired or invalid."}), 400

        state['granularity'] = granularity
        state['state'] = 'awaiting_output_format'
        save_session(session_id, state)
        
        return jsonify({
            "success": True,
            "message": "How would you like the results displayed?",
            "options": [
                {"id": "label", "text": "Label Only"},
                {"id": "label_category", "text": "Label + Category"}
            ]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chat/select-output-format', methods=['POST'])
def select_output_format():
    """Handle output format selection and prepare for analysis"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        output_format = data.get('output_format')
        
        state = get_session(session_id)
        if not state:
            return jsonify({"error": "Invalid session"}), 400
        
        state['output_format'] = output_format
        
        # Check if single text analysis
        if state.get('state') == 'awaiting_output_format_single':
            save_session(session_id, state)
            return analyze_single_text_internal(session_id, output_format)
        
        # For file uploads, prepare content_to_analyze
        file_id = state.get('file_id')
        if not file_id:
            return jsonify({"error": "No file uploaded"}), 400
        
        file_data = get_file(file_id)
        if not file_data:
            return jsonify({"error": "File data missing"}), 400

        granularity = state.get('granularity', 'sectional')
        content_type = state.get('content_type')
        
        # Special handling for "all" - analyze entire document as one
        if content_type == 'all':
            # Combine everything into one comprehensive analysis
            all_text = file_data.get('text_content', '')
            all_tables = file_data.get('tables_jsonl', [])
            all_figures = file_data.get('figures_jsonl', [])
            
            combined_summary = f"""Full Document Analysis:

Text Content (first 2000 characters):
{all_text[:2000]}

Number of Tables: {len(all_tables)}
Number of Figures: {len(all_figures)}

This represents the complete document including all text sections, tables, and figures for a comprehensive bias assessment."""
            
            content_to_analyze = [{
                "type": "all",
                "content": combined_summary,
                "caption": "Complete Document",
                "page": None
            }]
        else:
            # Normal flow: chunk by granularity and filter by type
            content_to_analyze = normalize_file_content(file_data, granularity)
            
            # Filter by content type
            if content_type and content_type != 'all':
                content_to_analyze = [c for c in content_to_analyze if c['type'] == content_type]

        if not content_to_analyze:
            return jsonify({
                "success": False,
                "error": "No content found to analyze. Try a different content type."
            }), 400

        # Save content_to_analyze back to file data
        file_data['content_to_analyze'] = content_to_analyze
        save_file(file_id, file_data)
        
        state['state'] = 'ready_for_analysis'
        save_session(session_id, state)
        
        return jsonify({
            "success": True,
            "message": f"Analyzing {len(content_to_analyze)} {'section' if len(content_to_analyze) == 1 else 'sections'}.",
            "total_items": len(content_to_analyze),
            "ready_for_batch": True
        })
        
    except Exception as e:
        print(f"Output format selection error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chat/analyze-batch', methods=['POST'])
def analyze_batch():
    """Process one chunk at a time for batch analysis"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        chunk_index = data.get('chunk_index', 0)

        state = get_session(session_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": "Session expired or invalid. Please start a new analysis."
            }), 400

        file_id = state.get('file_id')
        if not file_id:
            return jsonify({"success": False, "error": "No file data"}), 400
        
        file_data = get_file(file_id)
        if not file_data:
            return jsonify({"success": False, "error": "File data missing"}), 400

        content_to_analyze = file_data.get('content_to_analyze', [])
        output_format = state.get('output_format', 'label_category')

        if chunk_index >= len(content_to_analyze):
            state['state'] = 'analysis_complete'
            save_session(session_id, state)
            return jsonify({
                "success": True,
                "completed": True,
                "message": "Analysis complete!",
                "results": state.get("results", []),
            })

        chunk_obj = content_to_analyze[chunk_index]

        # Process chunk based on type
        chunk_type = chunk_obj["type"]
        content_for_model = chunk_obj["content"]
        
        result_text = detect_bias_with_model(content_for_model, output_format)

        # Extract label and category for logging
        try:
            label, category = extract_label_category(result_text)
        except Exception:
            label, category = "Unknown", "Unknown"

        # Create preview
        if chunk_type == "text":
            content_preview = str(content_for_model)[:120]
        elif chunk_type == "all":
            content_preview = "Complete Document (all content)"
        else:
            content_preview = chunk_obj.get("caption", f"{chunk_type} (no caption)")

        print(f"{chunk_index + 1} | {content_preview} | Label={label}, Category={category}")

        result = {
            "chunk_number": chunk_index + 1,
            "type": chunk_type,
            "content_preview": content_preview,
            "result": result_text,
            "error": False
        }

        # Save result to session
        if "results" not in state:
            state["results"] = []
        state["results"].append(result)
        save_session(session_id, state)

        is_complete = chunk_index + 1 == len(content_to_analyze)
        
        response_payload = {
            "success": True,
            "completed": is_complete,
            "result": result,
            "progress": {
                "current": chunk_index + 1,
                "total": len(content_to_analyze)
            }
        }

        if is_complete:
            state['state'] = 'analysis_complete'
            save_session(session_id, state)
            response_payload.update({
                "question": "What would you like to do next?",
                "options": [
                    {"id": "upload_file", "text": "Upload New File"},
                    {"id": "single_text", "text": "Analyze Text"},
                    {"id": "done", "text": "End Session"}
                ]
            })

        return jsonify(response_payload)

    except Exception as e:
        print(f"Batch analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chat/single-text', methods=['POST'])
def handle_single_text():
    """Handle single text input"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        state = get_session(session_id)

        text = data.get('text', '').strip()
        
        if not state:
            return jsonify({
                "success": False,
                "error": "Session expired or invalid. Please start a new analysis."
            }), 400
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        word_count = len(text.split())
        if word_count > MAX_TEXT_WORDS:
            return jsonify({
                "success": False,
                "error": f"Input exceeds {MAX_TEXT_WORDS} words. Please enter 200 words or fewer."
            }), 400
        
        state['single_text'] = text
        state['state'] = 'awaiting_output_format_single'
        save_session(session_id, state)
        
        return jsonify({
            "success": True,
            "message": "Text received. How would you like the results displayed?",
            "options": [
                {"id": "label", "text": "Label Only"},
                {"id": "label_category", "text": "Label + Category"}
            ]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def analyze_single_text_internal(session_id, output_format="label_category"):
    """Internal function to analyze single text"""
    state = get_session(session_id)

    if not state:
        return jsonify({"success": False, "error": "Invalid session"}), 400

    text = state.get('single_text', '')
    if not text:
        return jsonify({"success": False, "error": "No text found"}), 400

    result = detect_bias_with_model(text, output_format)

    try:
        label, category = extract_label_category(result)
    except Exception:
        label, category = "Unknown", "Unknown"

    preview = text[:50] + "..." if len(text) > 50 else text
    print(f"1 | {preview} | Predicted: Label={label}, Category={category}")

    state['state'] = 'single_analysis_complete'
    save_session(session_id, state)

    return jsonify({
        "success": True,
        "result": result,
        "message": "Analysis complete!",
        "question": "What would you like to do next?",
        "options": [
            {"id": "upload_file", "text": "Upload File"},
            {"id": "single_text", "text": "Analyze Text"},
            {"id": "done", "text": "End Session"}
        ]
    })


@app.route('/api/chat/continue', methods=['POST'])
def handle_continue():
    """Handle user's choice to continue or end"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        choice = data.get('choice')
        
        state = get_session(session_id)
        if not state:
            return jsonify({
                "success": False,
                "error": "Session expired or invalid. Please start a new analysis."
            }), 400
                        
        if choice == 'upload_file':
            state['state'] = 'awaiting_file'
            state['file_id'] = None
            save_session(session_id, state)
            return jsonify({
                "success": True,
                "message": "Please upload a new PDF file:",
                "file_upload_enabled": True
            })

        elif choice == 'single_text':
            state['state'] = 'single_text_ready'
            save_session(session_id, state)
            return jsonify({
                "success": True,
                "message": "Please enter the text you'd like to analyze:",
                "text_input_enabled": True
            })

        elif choice == 'done':
            delete_session(session_id)
            return jsonify({
                "success": True,
                "message": "Thank you for using the Coronary Artery Disease Bias Detector!",
                "conversation_ended": True
            })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def serve_react(path):
    if path != "" and os.path.exists("static/" + path):
        return send_from_directory("static", path)
    return send_from_directory("static", "index.html")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "fine_tuned_model": bool(FINE_TUNED_MODEL_ID),
        "ocr_model": bool(OCR_MODEL_ID),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)