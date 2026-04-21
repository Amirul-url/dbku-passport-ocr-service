import os
import re
import uuid

from django.conf import settings
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


COUNTRY_CODE_MAP = {
    "MYS": "Malaysian",
    "JPN": "Japanese",
    "IDN": "Indonesian",
    "SGP": "Singaporean",
    "THA": "Thai",
    "BRN": "Bruneian",
    "PHL": "Filipino",
    "VNM": "Vietnamese",
    "CHN": "Chinese",
    "KOR": "South Korean",
    "PRK": "North Korean",
    "IND": "Indian",
    "PAK": "Pakistani",
    "BGD": "Bangladeshi",
    "LKA": "Sri Lankan",
    "NPL": "Nepalese",
    "MMR": "Myanmar",
    "KHM": "Cambodian",
    "LAO": "Lao",
    "USA": "American",
    "CAN": "Canadian",
    "GBR": "British",
    "AUS": "Australian",
    "NZL": "New Zealander",
    "DEU": "German",
    "FRA": "French",
    "ITA": "Italian",
    "ESP": "Spanish",
    "PRT": "Portuguese",
    "NLD": "Dutch",
    "BEL": "Belgian",
    "CHE": "Swiss",
    "AUT": "Austrian",
    "SWE": "Swedish",
    "NOR": "Norwegian",
    "DNK": "Danish",
    "FIN": "Finnish",
    "IRL": "Irish",
    "POL": "Polish",
    "CZE": "Czech",
    "TUR": "Turkish",
    "SAU": "Saudi",
    "ARE": "Emirati",
    "QAT": "Qatari",
    "KWT": "Kuwaiti",
    "OMN": "Omani",
    "BHR": "Bahraini",
    "EGY": "Egyptian",
    "ZAF": "South African",
    "BRA": "Brazilian",
    "MEX": "Mexican",
    "RUS": "Russian",
}


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


def normalize_mrz_text(text):
    if not text:
        return ""

    text = text.upper()
    text = text.replace("«", "<").replace("‹", "<").replace("〈", "<")
    text = text.replace(" ", "")
    text = text.replace("|", "I")
    text = re.sub(r"[^A-Z0-9<\n]", "", text)
    return text


def find_mrz_lines(raw_text):
    if not raw_text:
        return []

    normalized = normalize_mrz_text(raw_text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    mrz_lines = []
    for line in lines:
        if len(line) >= 25 and "<" in line:
            mrz_lines.append(line)

    return mrz_lines


def clean_name_piece(value):
    if not value:
        return ""

    value = value.replace("<", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value.title()


def split_name_from_mrz(surname_part, given_part):
    surname = clean_name_piece(surname_part)
    given = clean_name_piece(given_part)

    return {
        "first_name": given,
        "last_name": surname,
        "full_name": " ".join(part for part in [given, surname] if part).strip(),
    }


def yyMMdd_to_iso(value):
    if not value or len(value) != 6 or not value.isdigit():
        return ""

    yy = int(value[0:2])
    mm = value[2:4]
    dd = value[4:6]

    # simple passport-style year handling
    # birth year: assume 1900s for larger yy, 2000s for smaller yy
    # expiry year: caller can override if needed, but this is good enough for UI
    year = 1900 + yy if yy >= 30 else 2000 + yy

    return f"{year:04d}-{mm}-{dd}"


def parse_two_line_mrz(raw_text):
    mrz_lines = find_mrz_lines(raw_text)
    if len(mrz_lines) < 2:
        return {}

    line1 = mrz_lines[0]
    line2 = mrz_lines[1]

    result = {
        "type": "P",
        "country_code": "",
        "nationality": "",
        "passport_number": "",
        "first_name": "",
        "last_name": "",
        "full_name": "",
        "date_of_birth": "",
        "date_of_issue": "",
        "date_of_expiry": "",
        "sex": "",
        "gender": "",
    }

    try:
        if line1.startswith("P<") and len(line1) >= 5:
            result["type"] = "P"
            result["country_code"] = line1[2:5]

            nationality = COUNTRY_CODE_MAP.get(result["country_code"], "")
            result["nationality"] = nationality

            name_block = line1[5:]
            if "<<" in name_block:
                surname_part, given_part = name_block.split("<<", 1)
                name_parts = split_name_from_mrz(surname_part, given_part)
                result.update(name_parts)

        if len(line2) >= 27:
            passport_number = re.sub(r"[^A-Z0-9]", "", line2[0:9]).replace("<", "")
            nationality_code = re.sub(r"[^A-Z]", "", line2[10:13])

            birth_raw = re.sub(r"[^0-9]", "", line2[13:19])
            sex_raw = line2[20:21].upper() if len(line2) > 20 else ""
            expiry_raw = re.sub(r"[^0-9]", "", line2[21:27])

            if passport_number:
                result["passport_number"] = passport_number

            if nationality_code and not result["country_code"]:
                result["country_code"] = nationality_code

            if nationality_code and not result["nationality"]:
                result["nationality"] = COUNTRY_CODE_MAP.get(nationality_code, "")

            if birth_raw:
                result["date_of_birth"] = yyMMdd_to_iso(birth_raw)

            if expiry_raw:
                result["date_of_expiry"] = yyMMdd_to_iso(expiry_raw)

            if sex_raw in ["M", "F", "X"]:
                result["sex"] = sex_raw
                result["gender"] = sex_raw

    except Exception:
        return result

    return result


def extract_visual_fallback(raw_text):
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    result = {
        "type": "P",
        "country_code": "",
        "nationality": "",
        "passport_number": "",
        "first_name": "",
        "last_name": "",
        "full_name": "",
        "date_of_birth": "",
        "date_of_issue": "",
        "date_of_expiry": "",
        "sex": "",
        "gender": "",
    }

    upper_lines = [line.upper() for line in lines]
    full_upper = "\n".join(upper_lines)

    passport_match = re.search(r"\b[A-Z]{1,2}\d{6,8}\b", full_upper)
    if passport_match:
        result["passport_number"] = passport_match.group(0)

    for code in COUNTRY_CODE_MAP.keys():
        if f" {code} " in f" {full_upper} ":
            result["country_code"] = code
            result["nationality"] = COUNTRY_CODE_MAP.get(code, "")
            break

    for i, line in enumerate(upper_lines):
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if ("NATIONALITY" in line or "COUNTRY CODE" in line) and next_line and not result["nationality"]:
            upper_next = next_line.upper().strip()
            if upper_next in COUNTRY_CODE_MAP:
                result["country_code"] = upper_next
                result["nationality"] = COUNTRY_CODE_MAP.get(upper_next, "")
            else:
                result["nationality"] = next_line.title()

        if ("SEX" in line or "GENDER" in line) and next_line and not result["sex"]:
            sex_val = next_line.strip().upper()[:1]
            if sex_val in ["M", "F", "X"]:
                result["sex"] = sex_val
                result["gender"] = sex_val

        if ("DATE OF BIRTH" in line or "BIRTH" in line) and next_line and not result["date_of_birth"]:
            result["date_of_birth"] = next_line

        if ("DATE OF ISSUE" in line or "ISSUE" in line) and next_line and not result["date_of_issue"]:
            result["date_of_issue"] = next_line

        if ("DATE OF EXPIRY" in line or "EXPIRY" in line or "EXPIRE" in line) and next_line and not result["date_of_expiry"]:
            result["date_of_expiry"] = next_line

        if ("SURNAME" in line or "LAST NAME" in line) and next_line and not result["last_name"]:
            result["last_name"] = next_line.title()

        if ("GIVEN NAME" in line or "FIRST NAME" in line) and next_line and not result["first_name"]:
            result["first_name"] = next_line.title()

    result["full_name"] = " ".join(
        part for part in [result["first_name"], result["last_name"]] if part
    ).strip()

    return result


def merge_extraction(raw_text):
    mrz = parse_two_line_mrz(raw_text)
    visual = extract_visual_fallback(raw_text)

    result = {
        "type": mrz.get("type") or visual.get("type") or "P",
        "country_code": mrz.get("country_code") or visual.get("country_code") or "",
        "nationality": mrz.get("nationality") or visual.get("nationality") or "",
        "passport_number": mrz.get("passport_number") or visual.get("passport_number") or "",
        "first_name": mrz.get("first_name") or visual.get("first_name") or "",
        "last_name": mrz.get("last_name") or visual.get("last_name") or "",
        "full_name": mrz.get("full_name") or visual.get("full_name") or "",
        "date_of_birth": mrz.get("date_of_birth") or visual.get("date_of_birth") or "",
        "date_of_issue": mrz.get("date_of_issue") or visual.get("date_of_issue") or "",
        "date_of_expiry": mrz.get("date_of_expiry") or visual.get("date_of_expiry") or "",
        "sex": mrz.get("sex") or visual.get("sex") or "",
        "gender": mrz.get("gender") or visual.get("gender") or "",
    }

    # fallback: if no split names but have full name, put full name into first_name
    if not result["first_name"] and not result["last_name"] and result["full_name"]:
        result["first_name"] = result["full_name"]

    return result


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
    fields = merge_extraction(raw_text)

    status = "auto-extracted"
    if not fields.get("passport_number"):
        status = "pending verification"

    original_image_url = request.build_absolute_uri(
        f"/media/passport_images/{original_filename}"
    )
    processed_image_url = (
        request.build_absolute_uri(f"/media/passport_processed/{processed_filename}")
        if processed_filename
        else ""
    )

    return JsonResponse({
        "message": "Passport scanned successfully",
        "status": status,
        "type": fields.get("type", "P"),
        "country_code": fields.get("country_code", ""),
        "passport_number": fields.get("passport_number", ""),
        "nationality": fields.get("nationality", ""),
        "first_name": fields.get("first_name", ""),
        "last_name": fields.get("last_name", ""),
        "full_name": fields.get("full_name", ""),
        "date_of_birth": fields.get("date_of_birth", ""),
        "sex": fields.get("sex", ""),
        "date_of_issue": fields.get("date_of_issue", ""),
        "date_of_expiry": fields.get("date_of_expiry", ""),
        "country": fields.get("nationality", ""),
        "expiry_date": fields.get("date_of_expiry", ""),
        "gender": fields.get("gender", fields.get("sex", "")),
        "raw_text": raw_text,
        "image_quality_note": "",
        "confidence_score": 0,
        "original_image_name": original_filename,
        "processed_image_name": processed_filename or "",
        "original_image_url": original_image_url,
        "processed_image_url": processed_image_url,
        "dynamic_fields": [],
    })