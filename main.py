from fastapi import FastAPI, File, UploadFile, HTTPException, staticfiles
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import io
from PIL import Image, ImageDraw, ImageFont
import logging
from enum import Enum
from typing import Optional
from pillow_heif import register_heif_opener
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


class ModelType(str, Enum):
    eccv16 = "eccv16"
    siggraph17 = "siggraph17"


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


@api_app.post("/colorize/")
async def colorize_image(
        file: UploadFile = File(...),
        model: ModelType = ModelType.siggraph17
):
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


@api_app.post("/colorize/compare")
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
app.mount("/", staticfiles.StaticFiles(directory="ui", html=True), name="ui")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
