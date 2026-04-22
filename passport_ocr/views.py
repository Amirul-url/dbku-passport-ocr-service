import os
import re
import threading
import uuid

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import pytesseract
from PIL import Image
from paddleocr import PaddleOCR


if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


PADDLE_OCR = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)
PADDLE_OCR_LOCK = threading.Lock()


COUNTRY_CODE_MAP = {
    "MYS": "Malaysia",
    "JPN": "Japan",
    "IDN": "Indonesia",
    "SGP": "Singapore",
    "THA": "Thailand",
    "BRN": "Brunei",
    "PHL": "Philippines",
    "VNM": "Vietnam",
    "CHN": "China",
    "KOR": "South Korea",
    "PRK": "North Korea",
    "IND": "India",
    "PAK": "Pakistan",
    "BGD": "Bangladesh",
    "LKA": "Sri Lanka",
    "NPL": "Nepal",
    "MMR": "Myanmar",
    "KHM": "Cambodia",
    "LAO": "Laos",
    "USA": "United States",
    "CAN": "Canada",
    "GBR": "United Kingdom",
    "AUS": "Australia",
    "NZL": "New Zealand",
    "DEU": "Germany",
    "FRA": "France",
    "ITA": "Italy",
    "ESP": "Spain",
    "PRT": "Portugal",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "CHE": "Switzerland",
    "AUT": "Austria",
    "SWE": "Sweden",
    "NOR": "Norway",
    "DNK": "Denmark",
    "FIN": "Finland",
    "IRL": "Ireland",
    "POL": "Poland",
    "CZE": "Czech Republic",
    "TUR": "Turkey",
    "SAU": "Saudi Arabia",
    "ARE": "United Arab Emirates",
    "QAT": "Qatar",
    "KWT": "Kuwait",
    "OMN": "Oman",
    "BHR": "Bahrain",
    "EGY": "Egypt",
    "ZAF": "South Africa",
    "BRA": "Brazil",
    "MEX": "Mexico",
    "RUS": "Russia",
}

GENERIC_PASSPORT_PATTERN = r"^[A-Z0-9]{6,12}$"

COUNTRY_PASSPORT_PATTERNS = {
    "JPN": r"^[A-Z]{2}\d{7}$",
    "KOR": r"^[A-Z]{1}[A-Z0-9]{8}$",
    "USA": r"^\d{9}$",
    "GBR": r"^\d{9}$",
    "IND": r"^[A-Z]{1}\d{7}$",
    "IDN": r"^[A-Z]{1,2}\d{6,8}$",
    "MYS": r"^[A-Z]{1}\d{8}$",
    "CHN": r"^[A-Z0-9]{8,9}$",
    "SGP": r"^[A-Z]\d{7}[A-Z]?$",
    "THA": r"^[A-Z]{1,2}\d{6,7}$",
}


def ensure_media_dirs():
    os.makedirs(os.path.join(settings.MEDIA_ROOT, "passport_images"), exist_ok=True)
    os.makedirs(os.path.join(settings.MEDIA_ROOT, "passport_processed"), exist_ok=True)


def country_code_to_name(code):
    code = (code or "").strip().upper()
    return COUNTRY_CODE_MAP.get(code, code if code else "Unknown")


def validate_passport_number_by_country(passport_number, country_code_or_name):
    passport_number = re.sub(r"[^A-Z0-9]", "", (passport_number or "").upper())
    country_value = (country_code_or_name or "").strip()

    if not passport_number:
        return False, "Passport number cannot be empty"

    if not re.match(GENERIC_PASSPORT_PATTERN, passport_number):
        return False, "Passport number format is invalid"

    country_code = ""
    upper_value = country_value.upper()

    if upper_value in COUNTRY_CODE_MAP:
        country_code = upper_value
    else:
        reverse_map = {v.upper(): k for k, v in COUNTRY_CODE_MAP.items()}
        country_code = reverse_map.get(upper_value, "")

    pattern = COUNTRY_PASSPORT_PATTERNS.get(country_code)
    if pattern and not re.match(pattern, passport_number):
        country_label = COUNTRY_CODE_MAP.get(country_code, country_code)
        return False, f"Invalid passport format for {country_label}"

    return True, ""


def fix_common_ocr_errors(text, mode="general"):
    if not text:
        return ""

    text = text.strip().upper()

    if mode == "mrz":
        text = text.replace(" ", "")
        text = text.replace("«", "<").replace("‹", "<").replace("〈", "<")
        text = text.replace("|", "I")
        text = re.sub(r"[^A-Z0-9<]", "", text)

        text = text.replace("KKKK", "<<<<")
        text = text.replace("KKK", "<<<")
        text = text.replace("KK", "<<")

        text = text.replace("5PN", "JPN")
        text = text.replace("2PN", "JPN")
        text = text.replace("7PN", "JPN")
        text = text.replace("25PN", "JPN")
        text = text.replace("MYS:", "MYS")
        text = text.replace("MYSMA", "MYSMA")
        text = text.replace("P<MYSMA", "P<MYSMA")
        text = text.replace("P<JPNNA", "P<JPNNA")
        text = text.replace("O<", "0<")
        text = text.replace("<K<", "<<")
        return text

    if mode == "passport":
        text = re.sub(r"[^A-Z0-9]", "", text)
        if not text:
            return ""

        text = (
            text.replace("O", "0")
            .replace("Q", "0")
            .replace("I", "1")
            .replace("L", "1")
            .replace("S", "5")
            .replace("B", "8")
        )
        return text

    if mode == "name":
        text = text.replace("0", "O")
        text = text.replace("1", "I")
        text = text.replace("5", "S")
        text = re.sub(r"[^A-Z< ]", "", text)
        return text

    return text


def normalize_display_date(value):
    if not value:
        return ""

    value = str(value).strip()
    if not value:
        return ""

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
    if m:
        return f"{m.group(3)}/{m.group(2)}/{m.group(1)}"

    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", value)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"

    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", value)
    if m:
        first = int(m.group(1))
        second = int(m.group(2))
        if first <= 12 and second <= 31:
            return f"{m.group(2)}/{m.group(1)}/{m.group(3)}"

    return value.replace("-", "/")


def is_reasonable_name(value):
    if not value:
        return False

    value = str(value).strip()
    if len(value) < 2:
        return False

    if re.search(r"\d", value):
        return False

    banned = [
        "passport",
        "nationality",
        "country",
        "date",
        "issue",
        "expiry",
        "birth",
        "sex",
        "male",
        "female",
        "identity",
    ]
    lower_value = value.lower()
    if any(word in lower_value for word in banned):
        return False

    cleaned = re.sub(r"[^A-Za-z\s'/-]", "", value).strip()
    return len(cleaned) >= 2


def clean_person_name(value):
    if not value:
        return ""

    value = value.replace("<", " ")
    value = re.sub(r"[^A-Za-z\s'/-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    if not is_reasonable_name(value):
        return ""

    return value


def rescue_mrz_lines(text):
    if not text:
        return []

    raw_lines = [x.strip().upper() for x in text.splitlines() if x.strip()]
    rescued = []

    for line in raw_lines:
        line = fix_common_ocr_errors(line, mode="mrz")
        line = re.sub(r"[^A-Z0-9<]", "", line)
        if len(line) >= 25 and "<" in line:
            rescued.append(line)

    return rescued


def mrz_char_value(c):
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - 55
    if c == "<":
        return 0
    return 0


def mrz_check_digit(data):
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        total += mrz_char_value(char) * weights[i % 3]
    return str(total % 10)


def split_passport_name_parts(country_code, surname_part, given_part):
    surname = clean_person_name(fix_common_ocr_errors(surname_part, mode="name"))
    given = clean_person_name(fix_common_ocr_errors(given_part, mode="name"))

    if not given and surname:
        parts = surname.split()
        if len(parts) > 1:
            surname = parts[0]
            given = " ".join(parts[1:])

    full_name = " ".join(x for x in [given, surname] if x).strip()

    return {
        "surname": surname.strip(),
        "given_name": given.strip(),
        "full_name": full_name,
    }


def resolve_passport_name_parts(extracted):
    first_name = (extracted.get("first_name") or extracted.get("given_name") or "").strip()
    last_name = (extracted.get("last_name") or extracted.get("surname") or "").strip()
    full_name = (extracted.get("full_name") or "").strip()

    if not first_name and not last_name and full_name:
        parts = full_name.split()
        if len(parts) == 1:
            first_name = parts[0]
            last_name = ""
        else:
            first_name = " ".join(parts[:-1]).strip()
            last_name = parts[-1].strip()

    if not full_name:
        full_name = " ".join(x for x in [first_name, last_name] if x).strip()

    return {
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name,
    }


def get_safe_raw_text(extracted):
    return (
        extracted.get("raw_text")
        or extracted.get("ocr_raw_text")
        or extracted.get("mrz_text")
        or ""
    ).strip()


def repair_mrz_line2(line2):
    if not line2:
        return ""

    line2 = fix_common_ocr_errors(line2, mode="mrz")
    line2 = line2.replace("5PN", "JPN")
    line2 = line2.replace("7PN", "JPN")
    line2 = line2.replace("25PN", "JPN")
    line2 = line2.replace("MYS", "MYS")

    first9 = line2[:9]
    rest = line2[9:]

    first9 = re.sub(r"[^A-Z0-9<]", "", first9)
    first9 = first9.replace("O", "0").replace("Q", "0")
    first9 = first9.replace("I", "1").replace("L", "1")

    return first9 + rest


def parse_two_line_passport_mrz(line1, line2, rescue_mode=False):
    try:
        line1 = fix_common_ocr_errors(line1, mode="mrz")
        line2 = repair_mrz_line2(line2)

        line1 = re.sub(r"[^A-Z0-9<]", "", line1)
        line2 = re.sub(r"[^A-Z0-9<]", "", line2)

        line1 = line1[:44].ljust(44, "<")
        line2 = line2[:44].ljust(44, "<")

        if not line1.startswith("P<"):
            return None

        issuing_country = line1[2:5]
        names_part = line1[5:]

        if "<<" in names_part:
            surname_part, given_part = names_part.split("<<", 1)
        else:
            repaired_names = names_part.replace("<K<", "<<")
            if "<<" not in repaired_names and "<" in repaired_names:
                repaired_names = repaired_names.replace("<", "<<", 1)

            if "<<" in repaired_names:
                surname_part, given_part = repaired_names.split("<<", 1)
            else:
                surname_part = repaired_names
                given_part = ""

        name_parts = split_passport_name_parts(
            country_code=issuing_country,
            surname_part=surname_part,
            given_part=given_part,
        )

        passport_raw = line2[0:9]
        passport_check = line2[9]
        nationality = line2[10:13]
        dob_raw = line2[13:19]
        dob_check = line2[19]
        gender_char = line2[20]
        expiry_raw = line2[21:27]
        expiry_check = line2[27]

        passport_number = fix_common_ocr_errors(
            passport_raw.replace("<", ""),
            mode="passport",
        )

        passport_valid = mrz_check_digit(passport_raw) == passport_check
        dob_valid = mrz_check_digit(dob_raw) == dob_check
        expiry_valid = mrz_check_digit(expiry_raw) == expiry_check

        def yymmdd_to_iso(raw, birth=False):
            if not raw or len(raw) != 6 or not raw.isdigit():
                return ""
            yy = int(raw[:2])
            mm = raw[2:4]
            dd = raw[4:6]
            if birth:
                year = 1900 + yy if yy >= 30 else 2000 + yy
            else:
                year = 2000 + yy if yy < 70 else 1900 + yy
            return f"{year:04d}-{mm}-{dd}"

        result = {
            "type": "P",
            "country_code": issuing_country,
            "nationality_code": nationality,
            "passport_number": passport_number,
            "surname": name_parts.get("surname", ""),
            "given_name": name_parts.get("given_name", ""),
            "full_name": name_parts.get("full_name", ""),
            "date_of_birth": yymmdd_to_iso(dob_raw, birth=True),
            "date_of_issue": "",
            "date_of_expiry": yymmdd_to_iso(expiry_raw, birth=False),
            "gender": gender_char if gender_char in ["M", "F", "X"] else "",
            "sex": gender_char if gender_char in ["M", "F", "X"] else "",
            "passport_valid": passport_valid,
            "dob_valid": dob_valid,
            "expiry_valid": expiry_valid,
            "mrz_text": f"{line1}\n{line2}",
        }

        result["nationality"] = country_code_to_name(nationality or issuing_country)
        return result
    except Exception:
        return None


def parse_mrz_rescue(text):
    mrz_lines = rescue_mrz_lines(text)
    if len(mrz_lines) < 2:
        return None

    best_pair = None
    best_score = -1

    for i in range(len(mrz_lines) - 1):
        l1 = mrz_lines[i]
        l2 = mrz_lines[i + 1]

        score = 0
        if l1.startswith("P<"):
            score += 2
        if len(l1) >= 35:
            score += 1
        if len(l2) >= 35:
            score += 1
        if "<" in l1 and "<" in l2:
            score += 1

        if score > best_score:
            best_score = score
            best_pair = (l1, l2)

    if not best_pair:
        return None

    return parse_two_line_passport_mrz(best_pair[0], best_pair[1], rescue_mode=True)


def rotate_image_keep_bounds(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def paddleocr_lines_from_image(image):
    lines = []

    with PADDLE_OCR_LOCK:
        result = PADDLE_OCR.ocr(image, cls=False)

    if not result:
        return lines

    try:
        for item in result:

            # FORMAT BARU (dict)
            if isinstance(item, dict):
                texts = item.get("rec_text", [])
                scores = item.get("rec_score", [])

                for i in range(len(texts)):
                    text = texts[i]
                    score = scores[i] if i < len(scores) else 0

                    if str(text).strip():
                        lines.append({
                            "text": str(text).strip(),
                            "score": float(score)
                        })

            # FORMAT LAMA (list)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, list) and len(sub) >= 2:
                        if isinstance(sub[1], list):
                            text = sub[1][0]
                            score = sub[1][1] if len(sub[1]) > 1 else 0
                        else:
                            continue

                        if str(text).strip():
                            lines.append({
                                "text": str(text).strip(),
                                "score": float(score)
                            })

    except Exception:
        pass

    return lines


def choose_best_orientation_by_ocr(image):
    best_angle = 0
    best_score = -1
    best_lines = []

    for angle in [0, 90, 180, 270]:
        rotated = rotate_image_keep_bounds(image, angle) if angle else image.copy()
        lines = paddleocr_lines_from_image(rotated)

        score = 0
        for line in lines:
            text = line.get("text", "")
            confidence = float(line.get("score", 0))
            score += confidence
            if "P<" in text.upper():
                score += 5
            if "<" in text:
                score += 2
            if len(text) >= 25:
                score += 1

        if score > best_score:
            best_score = score
            best_angle = angle
            best_lines = lines

    return best_angle, best_lines


def extract_mrz_data(text):
    text = fix_common_ocr_errors(text, mode="mrz")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    normalized = []
    for line in lines:
        clean_line = re.sub(r"[^A-Z0-9<]", "", line.upper())
        if len(clean_line) >= 25:
            normalized.append(clean_line)

    for i in range(len(normalized) - 1):
        parsed = parse_two_line_passport_mrz(normalized[i], normalized[i + 1])
        if parsed:
            return parsed

    rescue = parse_mrz_rescue(text)
    if rescue:
        return rescue

    return None


def crop_mrz_region(image):
    h, w = image.shape[:2]
    start_y = int(h * 0.72)
    if start_y < 0:
        start_y = 0
    return image[start_y:h, 0:w]


def get_mrz_variants(image):
    mrz = crop_mrz_region(image)
    unique = uuid.uuid4().hex

    variants = []
    variants.append((f"mrz_raw_{unique}.jpg", mrz))

    gray = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY) if len(mrz.shape) == 3 else mrz
    variants.append((f"mrz_gray_{unique}.jpg", gray))

    big = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    variants.append((f"mrz_big_{unique}.jpg", big))

    _, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append((f"mrz_thresh_{unique}.jpg", thresh))

    inv = cv2.bitwise_not(thresh)
    variants.append((f"mrz_inv_{unique}.jpg", inv))

    return variants


def image_to_tesseract_text(image):
    try:
        pil_image = Image.fromarray(image)
        return pytesseract.image_to_string(pil_image)
    except Exception:
        return ""


def merge_unique_lines(text_blocks):
    seen = set()
    merged = []

    for block in text_blocks:
        if not block:
            continue

        for line in block.splitlines():
            clean = line.strip()
            if not clean:
                continue
            key = clean.upper()
            if key in seen:
                continue
            seen.add(key)
            merged.append(clean)

    return "\n".join(merged).strip()


def extract_basic_visual_fields(text):
    result = {
        "country_code": "",
        "passport_number": "",
        "nationality": "",
        "full_name": "",
        "date_of_birth": "",
        "date_of_issue": "",
        "date_of_expiry": "",
        "sex": "",
    }

    if not text:
        return result

    upper = text.upper()

    country_match = re.search(r"\b([A-Z]{3})\b", upper)
    if country_match:
        code = country_match.group(1)
        if code in COUNTRY_CODE_MAP:
            result["country_code"] = code
            result["nationality"] = country_code_to_name(code)

    passport_match = re.search(r"\b[A-Z]{1,2}\d{6,8}\b", upper)
    if passport_match:
        result["passport_number"] = fix_common_ocr_errors(passport_match.group(0), mode="passport")

    dob_match = re.search(r"(\d{2}\s+[A-Z]{3}\s+\d{4})", upper)
    if dob_match:
        result["date_of_birth"] = dob_match.group(1)

    date_matches = re.findall(r"(\d{2}\s+[A-Z]{3}\s+\d{4})", upper)
    if len(date_matches) >= 2:
        result["date_of_issue"] = date_matches[-2]
        result["date_of_expiry"] = date_matches[-1]

    if "MALE" in upper:
        result["sex"] = "M"
    elif "FEMALE" in upper:
        result["sex"] = "F"
    elif re.search(r"\bP[- ]?F\b", upper):
        result["sex"] = "F"
    elif re.search(r"\bP[- ]?M\b", upper):
        result["sex"] = "M"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        up = line.upper()
        if "NAME" in up and idx + 1 < len(lines):
            next_line = clean_person_name(lines[idx + 1])
            if next_line:
                result["full_name"] = next_line
                break

    return result


def process_passport_ocr(original_path, processed_path):
    image = cv2.imread(original_path)
    if image is None:
        return {
            "status": "pending verification",
            "raw_text": "",
            "image_quality_note": "Unable to read image",
            "confidence_score": 0,
            "detected_rotation_angle": 0,
        }

    best_angle, best_lines = choose_best_orientation_by_ocr(image)
    rotated = rotate_image_keep_bounds(image, best_angle) if best_angle else image

    cv2.imwrite(processed_path, rotated)

    full_image_text = "\n".join(
        [line.get("text", "") for line in best_lines if line.get("text", "").strip()]
    ).strip()

    mrz_text_blocks = []
    for _, variant in get_mrz_variants(rotated):
        variant_lines = paddleocr_lines_from_image(variant)
        if variant_lines:
            mrz_text_blocks.append(
                "\n".join([line.get("text", "") for line in variant_lines if line.get("text", "").strip()])
            )

        tess_text = image_to_tesseract_text(variant)
        if tess_text:
            mrz_text_blocks.append(tess_text)

    merged_mrz_text = merge_unique_lines(mrz_text_blocks)
    extracted = extract_mrz_data(merged_mrz_text) or {}

    if not extracted:
        tess_full_text = image_to_tesseract_text(rotated)
        combined_text = merge_unique_lines([merged_mrz_text, full_image_text, tess_full_text])
        extracted = extract_mrz_data(combined_text) or {}
        raw_text = combined_text
    else:
        raw_text = merge_unique_lines([merged_mrz_text, full_image_text])

    if not extracted:
        visual = extract_basic_visual_fields(raw_text)
        extracted = {
            "type": "P",
            "country_code": visual.get("country_code", ""),
            "passport_number": visual.get("passport_number", ""),
            "surname": "",
            "given_name": "",
            "full_name": visual.get("full_name", ""),
            "date_of_birth": visual.get("date_of_birth", ""),
            "date_of_issue": visual.get("date_of_issue", ""),
            "date_of_expiry": visual.get("date_of_expiry", ""),
            "nationality": visual.get("nationality", ""),
            "gender": visual.get("sex", ""),
            "sex": visual.get("sex", ""),
        }

    extracted["raw_text"] = raw_text
    extracted["image_quality_note"] = ""
    extracted["confidence_score"] = 0
    extracted["detected_rotation_angle"] = best_angle
    extracted["status"] = "auto-extracted" if extracted.get("passport_number") else "pending verification"

    if extracted.get("nationality_code") and not extracted.get("nationality"):
        extracted["nationality"] = country_code_to_name(extracted.get("nationality_code"))

    if extracted.get("country_code") and not extracted.get("nationality"):
        extracted["nationality"] = country_code_to_name(extracted.get("country_code"))

    return extracted


def build_universal_passport_fields(extracted):
    name_parts = resolve_passport_name_parts(extracted)
    raw_text = get_safe_raw_text(extracted)

    country_code = (
        extracted.get("country_code")
        or extracted.get("nationality_code")
        or ""
    ).strip().upper()

    nationality = (
        extracted.get("nationality")
        or country_code_to_name(country_code)
        or ""
    ).strip()

    passport_number = fix_common_ocr_errors(
        extracted.get("passport_number", ""),
        mode="passport",
    )

    full_name = name_parts.get("full_name", "").strip()

    if not full_name and extracted.get("full_name"):
        full_name = clean_person_name(extracted.get("full_name", ""))

    if not name_parts.get("first_name") and full_name:
        parts = full_name.split()
        if len(parts) == 1:
            first_name = parts[0]
            last_name = ""
        else:
            last_name = parts[-1]
            first_name = " ".join(parts[:-1])
    else:
        first_name = name_parts.get("first_name", "").strip()
        last_name = name_parts.get("last_name", "").strip()

    return {
        "type": (extracted.get("type") or "P").strip(),
        "country_code": country_code,
        "passport_number": passport_number,
        "nationality": nationality,
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name,
        "date_of_birth": extracted.get("date_of_birth", "").strip(),
        "sex": (extracted.get("sex") or extracted.get("gender") or "").strip(),
        "date_of_issue": extracted.get("date_of_issue", "").strip(),
        "date_of_expiry": (
            extracted.get("date_of_expiry")
            or extracted.get("expiry_date")
            or ""
        ).strip(),
        "custom_fields": extracted.get("custom_fields", []),
        "raw_text": raw_text,
        "status": (extracted.get("status") or "pending verification").strip(),
        "confidence_score": extracted.get("confidence_score", 0),
        "image_quality_note": extracted.get("image_quality_note", ""),
        "detected_rotation_angle": extracted.get("detected_rotation_angle", 0),
    }


@csrf_exempt
def extract_passport(request):
    if request.headers.get("X-API-KEY") != settings.PASSPORT_OCR_API_KEY:
        return JsonResponse({"error": "Unauthorized"}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    if "image" not in request.FILES:
        return JsonResponse({"error": "Please choose a passport image first"}, status=400)

    try:
        ensure_media_dirs()

        image = request.FILES["image"]
        ext = os.path.splitext(image.name)[1].lower() or ".jpg"
        unique_id = uuid.uuid4().hex

        original_filename = f"passport_{unique_id}{ext}"
        processed_filename = f"processed_{unique_id}.jpg"

        original_path = os.path.join(settings.MEDIA_ROOT, "passport_images", original_filename)
        processed_path = os.path.join(settings.MEDIA_ROOT, "passport_processed", processed_filename)

        with open(original_path, "wb+") as file_obj:
            for chunk in image.chunks():
                file_obj.write(chunk)

        extracted = process_passport_ocr(original_path, processed_path)
        ui_result = build_universal_passport_fields(extracted)

        final_status = ui_result.get("status", "pending verification")
        if not ui_result.get("passport_number"):
            final_status = "pending verification"

        return JsonResponse({
            "message": "Passport scanned successfully",
            "type": ui_result.get("type", "P"),
            "country_code": ui_result.get("country_code", ""),
            "passport_number": ui_result.get("passport_number", ""),
            "nationality": ui_result.get("nationality", ""),
            "first_name": ui_result.get("first_name", ""),
            "last_name": ui_result.get("last_name", ""),
            "full_name": ui_result.get("full_name", ""),
            "date_of_birth": normalize_display_date(ui_result.get("date_of_birth", "")),
            "sex": ui_result.get("sex", ""),
            "date_of_issue": normalize_display_date(ui_result.get("date_of_issue", "")),
            "date_of_expiry": normalize_display_date(ui_result.get("date_of_expiry", "")),
            "custom_fields": ui_result.get("custom_fields", []),
            "raw_text": ui_result.get("raw_text", ""),
            "status": final_status,
            "confidence_score": ui_result.get("confidence_score", 0),
            "image_quality_note": ui_result.get("image_quality_note", ""),
            "detected_rotation_angle": ui_result.get("detected_rotation_angle", 0),
            "original_image_name": original_filename,
            "processed_image_name": processed_filename,
            "original_image_url": request.build_absolute_uri(
                f"{settings.MEDIA_URL}passport_images/{original_filename}"
            ),
            "processed_image_url": request.build_absolute_uri(
                f"{settings.MEDIA_URL}passport_processed/{processed_filename}"
            ),
        })

    except Exception as e:
        return JsonResponse({
            "error": str(e),
            "status": "pending verification",
        }, status=500)