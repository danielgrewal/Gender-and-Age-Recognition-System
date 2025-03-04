from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
#from app.filter_detector import FilterDetector

app = FastAPI()

@app.post("/api/process_frame")
async def process_frame(request: Request):
    return JSONResponse({"message": "Frame processed successfully"})
    request_data = request.json()
    # image_data = 
    

    image_data = request.get("image")
    
    if not image_data:
        return JSONResponse ({"error": "No image data provided."}, status_code = 400)
        
    detector = FilterDetector(image_data)
    filter_removed, img = detector.remove_filter()
    
    
    return JSONResponse({"message": "Frame processed successfully"})

        


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)