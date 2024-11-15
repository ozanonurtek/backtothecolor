from fastapi import FastAPI, File, UploadFile, HTTPException, staticfiles, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional
from middleware.headers import CustomHeaderMiddleware
from colorizer.service import ColorizationService, ModelType
from security.serializer import Serializer
from PIL import Image, ImageDraw, ImageFont
import io

import torch
import numpy as np
import logging
import imghdr

from pathlib import Path

serializer = Serializer()

# Initialize FastAPI app
api_app = FastAPI(
    title="Image Colorization API",
    description="Convert black and white images to color using ECCV16 and SIGGRAPH17 models",
    version="1.0.0"
)

i18n = {
  "en": {
    "title": "backtothecolor - Add Color to Your Old Photos with AI",
    "description": "Add color to your old black and white photos using our advanced AI colorization technology. Bring your memories back to life!",
    "appDescription": "add color to your black and white images using AI",
    "dragAndDropText": "drag and drop an image here or click to select",
    "colorizingText": "Colorizing... This may take up to 30 seconds...",
    "footerMadeWithText": "made with ❤️ by ",
    "footerOnurLinkText": "ozanonurtek. ",
    "footerGuzelyaliText": "in guzelyalı",
    "footerNoDataStoredText": "no data is stored on our servers"
  },
  "tr": {
    "title": "backtothecolor - Yapay Zeka ile Eski Fotoğraflarınıza Renk Ekleyin",
    "description": "Gelişmiş yapay zeka renklendirme teknolojimizi kullanarak eski siyah beyaz fotoğraflarınıza renk ekleyin. Anılarınızı hayata döndürün!",
    "appDescription": "Yapay zeka kullanarak siyah beyaz fotoğraflarınıza renk ekleyin",
    "dragAndDropText": "Buraya bir resim sürükleyip bırakın veya seçmek için tıklayın",
    "colorizingText": "Renklendirme... Bu 30 saniye sürebilir...",
    "footerMadeWithText": "Guzelyalı'da ❤️ ile ",
    "footerOnurLinkText": "ozanonurtek",
    "footerGuzelyaliText": " tarafından yapıldı.",
    "footerNoDataStoredText": "sunucularımızda veri saklanmaz"
  }
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the colorization service
colorizer_service = ColorizationService()


@api_app.post("/colorize/{identifier}")
async def colorize_image(
        identifier,
        request: Request,
        file: UploadFile = File(...),
        model: ModelType = ModelType.siggraph17
):
    try:
        x_real_ip = request.state.x_real_ip
        original = serializer.verify_token(identifier)
        if original != x_real_ip:
            raise HTTPException(400, "Unsupported identifier.")

        # Get request headers
        headers = dict(request.headers)
        logger.info(f"Request headers: {headers}")
        # Read file content
        content = await file.read()

        # Detect image format
        image_format = colorizer_service.detect_image_format(content)
        if not image_format:
            raise HTTPException(400, "Unsupported image format. Please upload a PNG, JPG, or HEIC image.")

        # Handle different image formats
        if image_format == 'heic':
            logger.info("Converting HEIC image to JPG")
            image = colorizer_service.convert_heic_to_jpg(content)
        else:
            image = Image.open(io.BytesIO(content)).convert('RGB')

        # Process image
        colorized_image = colorizer_service.process_image(image, model)

        # Prepare response
        output_buffer = io.BytesIO()
        colorized_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        return Response(
            content=output_buffer.getvalue(),
            media_type="image/png"
        )

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(500, f"Error processing image: {str(e)}")


#@api_app.post("/colorize/compare")
async def colorize_image_compare(file: UploadFile = File(...)):
    """
    Process an image with both models and return both results in a side-by-side comparison
    """
    try:
        # Read file content
        content = await file.read()

        # Detect image format
        image_format = colorizer_service.detect_image_format(content)
        if not image_format:
            raise HTTPException(400, "Unsupported image format. Please upload a PNG, JPG, or HEIC image.")

        # Handle different image formats
        if image_format == 'heic':
            logger.info("Converting HEIC image to JPG")
            image = colorizer_service.convert_heic_to_jpg(content)
        else:
            image = Image.open(io.BytesIO(content)).convert('RGB')

        # Process with both models
        eccv16_result = colorizer_service.process_image(image, ModelType.eccv16)
        siggraph17_result = colorizer_service.process_image(image, ModelType.siggraph17)

        # Create comparison image
        width = max(eccv16_result.width, siggraph17_result.width)
        total_width = width * 2
        height = max(eccv16_result.height, siggraph17_result.height)

        comparison = Image.new('RGB', (total_width, height))
        comparison.paste(eccv16_result, (0, 0))
        comparison.paste(siggraph17_result, (width, 0))

        # Prepare response
        output_buffer = io.BytesIO()
        comparison.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        return Response(
            content=output_buffer.getvalue(),
            media_type="image/png"
        )

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(500, f"Error processing image: {str(e)}")


@api_app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": ["eccv16", "siggraph17"],
        "supported_formats": ["jpg", "jpeg", "png", "heic"]
    }


app = FastAPI(title="main app")

app.mount("/api", api_app)
# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)
app.add_middleware(CustomHeaderMiddleware)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bactothecolor.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    headers = dict(request.headers)
    x_real_ip = request.state.x_real_ip
    lang = request.state.lang

    identifier = serializer.generate_token(x_real_ip)
    return templates.TemplateResponse(
        request=request, name="index.html", context={"identifier": identifier, "language": i18n[lang], "lang": lang}
    )

@app.get("/privacy", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="privacy.html"
    )

@app.get("/ads.txt", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="ads.txt"
    )

@app.get("/robots.txt", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="robots.txt"
    )

@app.get("/sitemap.xml", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="sitemap.xml"
    )