import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query
from dotenv import load_dotenv
from clarifai.client.model import Model
from clarifai.errors import ApiError as ClarifaiException

# Load .env
load_dotenv()

app = FastAPI(
    title="Clarifai Image Moderation (Scoring Only)",
    description="Endpoint trả điểm/phán đoán ảnh có phản cảm hay không, KHÔNG block tự động."
)

# Cấu hình
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY")
MODEL_URL = "https://clarifai.com/clarifai/main/models/moderation-recognition"
DEFAULT_UNSAFE_THRESHOLD = float(os.getenv("UNSAFE_THRESHOLD", 0.8))
BLOCKING_LABELS = ['suggestive', 'gore', 'drugs', 'hate', 'unsafe']

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}

clarifai_model = None
DEBUG = os.getenv("MODERATION_DEBUG", "false").lower() in ("1", "true", "yes")

# Khởi tạo Clarifai client (không crash nếu thiếu key)
try:
    if not CLARIFAI_API_KEY:
        print("❌ CLARIFAI_API_KEY chưa thiết lập. Service sẽ trả 503 cho các request.")
    else:
        clarifai_model = Model(MODEL_URL, pat=CLARIFAI_API_KEY)
        print("✅ Clarifai client khởi tạo thành công.")
except Exception as e:
    print(f"❌ Lỗi khởi tạo Clarifai client: {e}")
    clarifai_model = None


def _normalize_label_name(name: str) -> str:
    if not name:
        return ""
    return name.strip().lower()


@app.post("/api/v1/image/moderation/score")
async def check_image_moderation_score(
    image: UploadFile = File(...),
    threshold: Optional[float] = Query(None, description="Optional override for unsafe threshold (0.0-1.0)")
):
    """
    Trả về JSON chứa:
    - is_offensive: bool (max_score >= threshold)
    - offensive_score: float (max score trong các label)
    - offensive_labels: [{name, score}, ...] (danh sách label đã normalized)
    - note: debug info nếu MODERATION_DEBUG bật
    """

    # 1) Service ready?
    if not clarifai_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Moderation service inactive. CLARIFAI_API_KEY missing or invalid."
        )

    # 2) Basic validation: mime & size
    content_type = image.content_type or ""
    if content_type not in ALLOWED_MIMES:
        await image.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {content_type}"
        )

    try:
        image_bytes = await image.read()
    except Exception as e:
        await image.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot read uploaded file: {e}"
        )
    await image.close()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File too large. Max {MAX_FILE_SIZE} bytes.")

    # 3) Call Clarifai (upstream errors -> 502)
    try:
        response = clarifai_model.predict_by_bytes(image_bytes, input_type="image")
    except ClarifaiException as e:
        if DEBUG:
            print(f"[ClarifaiException] {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Error calling Clarifai moderation service.")
    except Exception as e:
        if DEBUG:
            print(f"[Exception] calling Clarifai: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error while processing image.")

    # 4) Parse response
    try:
        outputs = getattr(response, "outputs", None)
        if not outputs:
            if DEBUG:
                print("Clarifai response has no outputs.")
            return {
                "is_offensive": False,
                "offensive_score": 0.0,
                "offensive_labels": [],
                "note": "Clarifai response invalid"
            }
        concepts = outputs[0].data.concepts
    except Exception as e:
        if DEBUG:
            print(f"Cannot parse Clarifai response: {e}")
        return {
            "is_offensive": False,
            "offensive_score": 0.0,
            "offensive_labels": [],
            "note": "Clarifai response parsing failed"
        }

    # 5) Build labels and compute max score
    offensive_labels: List[dict] = []
    max_score = 0.0

    for concept in concepts:
        name = getattr(concept, "name", "") or getattr(concept, "id", "")
        score = getattr(concept, "value", None)
        if score is None:
            score = getattr(concept, "score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0

        normalized = _normalize_label_name(name)
        offensive_labels.append({"name": normalized, "score": score})
        if score > max_score:
            max_score = score

    # 6) Decide offensive boolean (use either provided threshold or default)
    use_threshold = DEFAULT_UNSAFE_THRESHOLD if threshold is None else float(threshold)
    is_offensive = max_score >= use_threshold

    # 7) Return structured result (always 200 for verdict)
    result = {
        "is_offensive": is_offensive,
        "offensive_score": max_score,
        "offensive_labels": offensive_labels,
        "threshold_used": use_threshold
    }
    if DEBUG:
        result["debug_note"] = "MODERATION_DEBUG enabled; full concepts returned."

    return result


# Run for development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))