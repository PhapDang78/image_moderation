import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from dotenv import load_dotenv
from clarifai.client.model import Model
# Sửa lỗi: Import ApiError và đặt bí danh (alias)
from clarifai.errors import ApiError as ClarifaiException 


# Load biến môi trường
load_dotenv()

app = FastAPI(
    title="Clarifai Image Moderation Service",
    description="Microservice sử dụng Clarifai để kiểm duyệt nội dung hình ảnh."
)

# --- Cấu hình Clarifai ---
# CHUẨN HÓA TÊN BIẾN: Sử dụng CLARIFAI_API_KEY (có chữ 'A' rõ ràng)
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY") 
MODEL_URL = "https://clarifai.com/clarifai/main/models/moderation-recognition"
UNSAFE_THRESHOLD = 0.8
BLOCKING_LABELS = ['suggestive', 'gore', 'drugs', 'hate', 'unsafe'] 

clarifai_model = None

# Khởi tạo Clarifai Model Client
# Logic này được thiết kế để không crash server khi khởi động nếu key bị lỗi
try:
    if not CLARIFAI_API_KEY:
        # Nếu key thiếu, Model sẽ là None, và endpoint sẽ trả về 503
        print("❌ Cảnh báo: CLARIFAI_API_KEY chưa được thiết lập. Service sẽ trả về 503.")
    else:
        # Khởi tạo Model bằng PAT
        clarifai_model = Model(MODEL_URL, pat=CLARIFAI_API_KEY)
        print("✅ Clarifai Model Client đã được khởi tạo thành công.")
    
except Exception as e:
    # Bắt lỗi nếu Model khởi tạo thất bại (ví dụ: key hết hạn hoặc lỗi kết nối)
    print(f"❌ Lỗi nghiêm trọng khi khởi tạo Clarifai Model: {e}")
    clarifai_model = None # Đặt lại Model là None để kích hoạt lỗi 503

# --- Endpoint Kiểm duyệt Hình ảnh ---
@app.post("/api/v1/image/moderation")
async def check_image_moderation(image: UploadFile = File(...)):
    # 1. Kiểm tra trạng thái Model: Trả về 503 nếu không khởi tạo được
    if not clarifai_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Moderation service is inactive. CLARIFAI_API_KEY may be missing or invalid."
        )

    # 2. Đọc dữ liệu ảnh
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể đọc file: {e}"
        )

    # 3. GỌI VÀ BẮT LỖI CLARIFI
    try:
        response = clarifai_model.predict_by_bytes(
            image_bytes, 
            input_type="image"
        )
    except ClarifaiException as e:
        print(f"Lỗi Clarifai API (502): {e}")
        # Trả về 502 Bad Gateway vì upstream API (Clarifai) bị lỗi
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, 
            detail=f"Lỗi trong quá trình gọi Clarifai API: {e}"
        )
    except Exception as e:
        print(f"Lỗi gọi API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi máy chủ nội bộ không xác định: {e}"
        )
        

    # 4. XỬ LÝ KẾT QUẢ VÀ RAISE HTTP 403
    
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