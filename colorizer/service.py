
from .base_color import *
from .eccv16 import *
from .siggraph17 import *
from .util import *
from PIL import Image, ImageDraw, ImageFont
from pillow_heif import register_heif_opener
from enum import Enum
import io
import imghdr

class ModelType(str, Enum):
    eccv16 = "eccv16"
    siggraph17 = "siggraph17"
    
class ColorizationService:
    def __init__(self):
        # Register HEIF opener with Pillow
        register_heif_opener()
        self.colorizer_eccv16 = eccv16(pretrained=True).eval()
        self.colorizer_siggraph17 = siggraph17(pretrained=True).eval()

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
    def convert_heic_to_jpg(self, heic_content: bytes) -> Image.Image:
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

    def detect_image_format(self, file_content: bytes) -> str:
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
