import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from dotenv import load_dotenv
from clarifai.client.model import Model
from clarifai.errors import ApiError as ClarifaiException
from typing import List

# Load biến môi trường
load_dotenv()

app = FastAPI(
    title="Clarifai Image Moderation Service",
    description="Microservice sử dụng Clarifai để kiểm duyệt nội dung hình ảnh."
)

# --- Cấu hình ---
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY")
MODEL_URL = "https://clarifai.com/clarifai/main/models/moderation-recognition"
# Ngưỡng tổng quát: block nếu có label với score >= UNSAFE_THRESHOLD
UNSAFE_THRESHOLD = float(os.getenv("UNSAFE_THRESHOLD", 0.8))
# Ngưỡng nhãn đặc biệt (nhãn trong BLOCKING_LABELS) có thể thấp hơn/mạnh hơn tùy policy
BLOCKING_LABELS = ['suggestive', 'gore', 'drugs', 'hate', 'unsafe']
BLOCKING_LABEL_THRESHOLD = float(os.getenv("BLOCKING_LABEL_THRESHOLD", 0.75))

# Hạn chế file upload
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}

clarifai_model = None
# Debug flag (environment)
DEBUG = os.getenv("MODERATION_DEBUG", "false").lower() in ("1", "true", "yes")

# Khởi tạo Clarifai Model Client (không crash server nếu thiếu key)
try:
    if not CLARIFAI_API_KEY:
        print("❌ Cảnh báo: CLARIFAI_API_KEY chưa được thiết lập. Service sẽ trả về 503.")
    else:
        clarifai_model = Model(MODEL_URL, pat=CLARIFAI_API_KEY)
        print("✅ Clarifai Model Client đã được khởi tạo thành công.")
except Exception as e:
    print(f"❌ Lỗi khi khởi tạo Clarifai Model: {e}")
    clarifai_model = None


def _normalize_label_name(name: str) -> str:
    if not name:
        return ""
    return name.strip().lower()


@app.post("/api/v1/image/moderation")
async def check_image_moderation(image: UploadFile = File(...)):
    # 0. Model ready?
    if not clarifai_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Moderation service is inactive. CLARIFAI_API_KEY may be missing or invalid."
        )

    # 1. Basic client-side validation: content type and size
    content_type = image.content_type or ""
    if content_type not in ALLOWED_MIMES:
        await image.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {content_type}"
        )

    # Read bytes and enforce size limit
    try:
        image_bytes = await image.read()
    except Exception as e:
        await image.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot read uploaded file: {e}"
        )

    # Close file object (free resources)
    await image.close()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is too large. Max allowed size is {MAX_FILE_SIZE} bytes."
        )

    # 2. Call Clarifai
    try:
        response = clarifai_model.predict_by_bytes(
            image_bytes,
            input_type="image"
        )
    except ClarifaiException as e:
        # Upstream error: treat as 502 (Bad Gateway)
        print(f"[ClarifaiException] {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Error calling Clarifai moderation service."
        )
    except Exception as e:
        print(f"[Exception] calling Clarifai: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing image."
        )

    # 3. Validate response shape
    try:
        outputs = getattr(response, "outputs", None)
        if not outputs or len(outputs) == 0:
            if DEBUG:
                print("Clarifai response has no outputs.")
            return {"is_unsafe": False, "message": "Clarifai response was invalid."}
        concepts = outputs[0].data.concepts
    except Exception as e:
        if DEBUG:
            print(f"Cannot parse Clarifai response: {e}")
        return {"is_unsafe": False, "message": "Clarifai response parsing failed."}

    # 4. Decision logic: gather labels and scores
    unsafe_labels: List[dict] = []
    max_score = 0.0
    is_unsafe = False

    for concept in concepts:
        # Depending on SDK, attributes may be concept.name and concept.value or concept.id
        name = getattr(concept, "name", "") or getattr(concept, "id", "")
        score = getattr(concept, "value", None)
        if score is None:
            # Try alternative field names
            score = getattr(concept, "score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0

        normalized = _normalize_label_name(name)
        unsafe_labels.append({"name": normalized, "score": score})
        if score > max_score:
            max_score = score

        # If any label in blocking list crosses its threshold => block
        if normalized in BLOCKING_LABELS and score >= BLOCKING_LABEL_THRESHOLD:
            is_unsafe = True

    # Also block if the global max_score exceeds UNSAFE_THRESHOLD
    if max_score >= UNSAFE_THRESHOLD:
        is_unsafe = True

    # Debug logging
    if DEBUG:
        print("Parsed concepts:", unsafe_labels)
        print("max_score:", max_score, "is_unsafe:", is_unsafe)

    # 5. Return or raise accordingly
    if is_unsafe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Image failed moderation check: Unsafe content detected.",
                "unsafe_score": max_score,
                "unsafe_labels": unsafe_labels
            }
        )

    return {
        "is_unsafe": False,
        "unsafe_score": max_score,
        "unsafe_labels": unsafe_labels
    }


# --- Chạy Server (cho phát triển) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))