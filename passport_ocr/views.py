import os
import uuid
from io import BytesIO

from django.conf import settings
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from PIL import Image
import pytesseract

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


def ensure_media_dirs():
    os.makedirs(settings.MEDIA_ROOT / "passport_images", exist_ok=True)
    os.makedirs(settings.MEDIA_ROOT / "passport_processed", exist_ok=True)


def save_uploaded_image(uploaded_file):
    ensure_media_dirs()
    ext = os.path.splitext(uploaded_file.name)[1].lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    full_path = settings.MEDIA_ROOT / "passport_images" / filename

    with open(full_path, "wb+") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    return filename, full_path


def preprocess_image(input_path):
    if cv2 is None or np is None:
        return None, None

    image = cv2.imread(str(input_path))
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    processed_filename = f"processed_{uuid.uuid4().hex}.jpg"
    processed_path = settings.MEDIA_ROOT / "passport_processed" / processed_filename
    cv2.imwrite(str(processed_path), gray)

    return processed_filename, processed_path


def run_basic_ocr(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception:
        return ""


def extract_simple_fields(raw_text):
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    full_name = ""
    passport_number = ""
    country = ""
    date_of_birth = ""
    expiry_date = ""
    gender = ""

    for i, line in enumerate(lines):
        upper = line.upper()

        if not passport_number and "PASSPORT" in upper and i + 1 < len(lines):
            candidate = lines[i + 1].replace(" ", "")
            if 6 <= len(candidate) <= 12:
                passport_number = candidate

        if not full_name and len(line.split()) >= 2 and not any(ch.isdigit() for ch in line):
            if len(line) <= 80:
                full_name = line

        if "NATIONALITY" in upper and i + 1 < len(lines):
            country = lines[i + 1]

        if "SEX" in upper and i + 1 < len(lines):
            gender = lines[i + 1]

        if "BIRTH" in upper and i + 1 < len(lines):
            date_of_birth = lines[i + 1]

        if "EXPIRY" in upper and i + 1 < len(lines):
            expiry_date = lines[i + 1]

    return {
        "full_name": full_name,
        "passport_number": passport_number,
        "country": country,
        "date_of_birth": date_of_birth,
        "expiry_date": expiry_date,
        "gender": gender,
    }


@csrf_exempt
def extract_passport(request):
    if request.headers.get("X-API-KEY") != settings.PASSPORT_OCR_API_KEY:
        return JsonResponse({"error": "Unauthorized"}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    if "image" not in request.FILES:
        return JsonResponse({"error": "Please upload passport image"}, status=400)

    uploaded_file = request.FILES["image"]

    original_filename, original_path = save_uploaded_image(uploaded_file)
    processed_filename, processed_path = preprocess_image(original_path)

    ocr_source = processed_path if processed_path else original_path
    raw_text = run_basic_ocr(ocr_source)
    fields = extract_simple_fields(raw_text)

    status = "auto-extracted"
    if not fields["full_name"] or not fields["passport_number"]:
        status = "pending verification"

    original_image_url = f"/media/passport_images/{original_filename}"
    processed_image_url = (
        f"/media/passport_processed/{processed_filename}" if processed_filename else ""
    )

    return JsonResponse({
        "message": "Passport scanned successfully",
        "status": status,
        "full_name": fields["full_name"],
        "passport_number": fields["passport_number"],
        "country": fields["country"],
        "date_of_birth": fields["date_of_birth"],
        "expiry_date": fields["expiry_date"],
        "gender": fields["gender"],
        "raw_text": raw_text,
        "image_quality_note": "",
        "confidence_score": 0,
        "original_image_name": original_filename,
        "processed_image_name": processed_filename or "",
        "original_image_url": original_image_url,
        "processed_image_url": processed_image_url,
    })