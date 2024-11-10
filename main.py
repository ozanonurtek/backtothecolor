from fastapi import FastAPI, File, UploadFile, HTTPException, staticfiles, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from enum import Enum
from typing import Optional
from pillow_heif import register_heif_opener
from PIL import Image, ImageDraw, ImageFont

import torch
import numpy as np
import io
import logging
import imghdr

from pathlib import Path

from colorizers import *

# Register HEIF opener with Pillow
register_heif_opener()

# Initialize FastAPI app
api_app = FastAPI(
    title="Image Colorization API",
    description="Convert black and white images to color using ECCV16 and SIGGRAPH17 models",
    version="1.0.0"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
serializer = URLSafeTimedSerializer("4XPuEHT4pw6I6Bo2nQl5SnAm")



class ModelType(str, Enum):
    eccv16 = "eccv16"
    siggraph17 = "siggraph17"

def generate_token(email, salt='EAs04WE5s53HlmHqic8BJPco'):
    return serializer.dumps(email, salt=salt)


def verify_token(token, salt='EAs04WE5s53HlmHqic8BJPco', max_age=86400):
    try:
        data = serializer.loads(token, salt=salt, max_age=max_age)
        return data
    except SignatureExpired:
        # Token is valid but expired
        return None
    except BadSignature:
        # Token is invalid
        return None



def detect_image_format(file_content: bytes) -> str:
    """
    Detect the image format from the file content
    """
    format_type = imghdr.what(None, file_content)
    if format_type:
        return format_type

    # Check for HEIC format (imghdr doesn't detect HEIC)
    if file_content.startswith(b'ftypheic') or file_content.startswith(b'ftypmif1'):
        return 'heic'

    return None


def convert_heic_to_jpg(heic_content: bytes) -> Image.Image:
    """
    Convert HEIC image to JPEG format
    """
    try:
        # Create a temporary BytesIO object to hold the HEIC data
        heic_buffer = io.BytesIO(heic_content)

        # Open and convert HEIC image
        image = Image.open(heic_buffer)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    except Exception as e:
        logger.error(f"Error converting HEIC image: {str(e)}")
        raise HTTPException(500, f"Error converting HEIC image: {str(e)}")


class ColorizationService:
    def __init__(self):
        logger.info("Initializing colorization models...")
        self.colorizer_eccv16 = eccv16(pretrained=True).eval()
        self.colorizer_siggraph17 = siggraph17(pretrained=True).eval()
        logger.info("Models initialized successfully")

    def process_image(self, image: Image.Image, model_type: ModelType) -> Image.Image:
        # Convert PIL Image to numpy array
        img = np.array(image)

        # Preprocess image
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

        # Select model
        if model_type == ModelType.eccv16:
            colorizer = self.colorizer_eccv16
        else:
            colorizer = self.colorizer_siggraph17

        # Colorize
        with torch.no_grad():
            output_ab = colorizer(tens_l_rs).cpu()

        # Postprocess
        colorized_image = postprocess_tens(tens_l_orig, output_ab)

        # Convert numpy array back to PIL Image
        colorized_img = Image.fromarray((colorized_image * 255).astype(np.uint8))

        return colorized_img


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
        headers = dict(request.headers)
        x_real_ip = headers.get("CF-Connecting-IP", "unknown")
        original = verify_token(identifier)
        if original != x_real_ip:
            raise HTTPException(400, "Unsupported identifier.")

        # Get request headers
        headers = dict(request.headers)
        logger.info(f"Request headers: {headers}")
        # Read file content
        content = await file.read()

        # Detect image format
        image_format = detect_image_format(content)
        if not image_format:
            raise HTTPException(400, "Unsupported image format. Please upload a PNG, JPG, or HEIC image.")

        # Handle different image formats
        if image_format == 'heic':
            logger.info("Converting HEIC image to JPG")
            image = convert_heic_to_jpg(content)
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
        image_format = detect_image_format(content)
        if not image_format:
            raise HTTPException(400, "Unsupported image format. Please upload a PNG, JPG, or HEIC image.")

        # Handle different image formats
        if image_format == 'heic':
            logger.info("Converting HEIC image to JPG")
            image = convert_heic_to_jpg(content)
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
    x_real_ip = headers.get("CF-Connecting-IP", "unknown")
    identifier = generate_token(x_real_ip)
    return templates.TemplateResponse(
        request=request, name="index.html", context={"identifier": identifier}
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

@app.get("/sw.js", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="sw.js"
    )