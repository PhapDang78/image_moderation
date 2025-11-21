# moderation_service.py (Phiên bản FIX lỗi ModuleNotFoundError)

import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from dotenv import load_dotenv
from clarifai.client.model import Model
# 🚨 ĐÃ SỬA: Import ClarifaiException theo cách mới trong SDK 11.x
from clarifai.errors import ApiError as ClarifaiException


# Load biến môi trường
load_dotenv()

app = FastAPI(
    title="Clarifai Image Moderation Service",
    description="Microservice sử dụng Clarifai để kiểm duyệt nội dung hình ảnh."
)

# --- Cấu hình Clarifai ---
CLARIFI_API_KEY = os.getenv("CLARIFAI_API_KEY")
MODEL_URL = "https://clarifai.com/clarifai/main/models/moderation-recognition"
UNSAFE_THRESHOLD = 0.8  # Ngưỡng an toàn (80%)

BLOCKING_LABELS = ['suggestive', 'gore', 'drugs', 'hate', 'unsafe'] 

clarifai_model = None

# Khởi tạo Clarifai Model Client
try:
    if not CLARIFI_API_KEY:
        # Kiểm tra API Key (có thể bỏ qua nếu bạn dùng PAT)
        raise ValueError("CLARIFI_API_KEY chưa được thiết lập.")
    
    clarifai_model = Model(MODEL_URL, pat=CLARIFI_API_KEY)
    print("✅ Clarifai Model Client đã được khởi tạo thành công.")
    
except Exception as e:
    print(f"❌ Lỗi khi khởi tạo Clarifai: {e}")
    pass


# --- Endpoint Kiểm duyệt Hình ảnh ---
@app.post("/api/v1/image/moderation")
async def check_image_moderation(image: UploadFile = File(...)):
    if not clarifai_model:
        return {"is_unsafe": False, "message": "Moderation service is inactive or failed to initialize."}

    # Đọc dữ liệu ảnh thành bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể đọc file: {e}"
        )

    # 1. GỌI VÀ BẮT LỖI CLARIFAI
    try:
        response = clarifai_model.predict_by_bytes(
            image_bytes, 
            input_type="image"
        )
    except ClarifaiException as e:
        # Bắt lỗi cụ thể từ Clarifai API
        print(f"Lỗi Clarifai API: {e}")
        return {"is_unsafe": False, "message": f"API check failed due to Clarifai error: {e}"}
    except Exception as e:
        # Bắt lỗi network hoặc I/O khác trong quá trình gọi API
        print(f"Lỗi gọi API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình gọi Clarifai API: {e}"
        )
        

    # 2. XỬ LÝ KẾT QUẢ VÀ RAISE HTTP 403
    
    if not response.outputs or not response.outputs[0].data.concepts:
        print("Phản hồi Clarifai không hợp lệ hoặc không có kết quả.")
        return {"is_unsafe": False, "message": "Clarifai response was invalid."}


    concepts = response.outputs[0].data.concepts
    
    is_unsafe = False
    max_score = 0.0
    unsafe_labels = []

    for concept in concepts:
        score = concept.value 
        name = concept.name
        
        if score >= UNSAFE_THRESHOLD:
            max_score = max(max_score, score)
            unsafe_labels.append({"name": name, "score": score})
            
            if name in BLOCKING_LABELS: 
                is_unsafe = True

    if is_unsafe:
        # FastAPI tự xử lý HTTPException này, không cần except
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


# --- Chạy Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)