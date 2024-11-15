from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class CustomHeaderMiddleware(BaseHTTPMiddleware):
    """
    CustomHeaderMiddleware
    """
    async def dispatch(self, request: Request, call_next):
        headers = dict(request.headers)
        headers = {k.lower(): v for k, v in headers.items()}
        x_real_ip = headers.get("cf-connecting-ip", "unknown")
        x_accept_language = self.preferred_language(headers.get("accept-language", "en"))

        request.state.x_real_ip = x_real_ip
        request.state.lang = x_accept_language

        response = await call_next(request)
        return response

    """
    Return the preferred language code. In case of a draw return the first one only.
    """
    def preferred_language(self, accept_language_header):
        accept_language_header = str(accept_language_header)
        res = self.parse_and_sort_accept_language(accept_language_header)
        if not res:
            return 'en'

        return res[0]['code']


    def parse_and_sort_accept_language(self,accept_language_header):
        """
        Parse and sort the Accept-Language header by quality values.

        :param accept_language_header: Quality-value string, e.g. 'en-US,en;q=0.9,fr-FR;q=0.8,fr;q=0.7,hu;q=0.6'
        :return: List of dictionaries with parsed language codes, sorted by quality in descending order.
        """
        result = []
        if not accept_language_header:
            return result
        # Split by commas to get individual language-quality pairs
        languages_and_qualities = accept_language_header.split(',')

        for language_quality_pair in languages_and_qualities:
            # Remove extra spaces
            language_quality_pair = language_quality_pair.strip()

            try:
                if ';' in language_quality_pair:
                    # Split into language and quality
                    lang, quality_part = language_quality_pair.split(';', 1)
                    lang = lang.strip()
                    quality = float(quality_part.split('=')[1].strip())  # Extract the quality value
                else:
                    lang = language_quality_pair.strip()
                    quality = 1.0  # Default quality if not specified

                result.append({
                    'code': lang.split('-')[0].lower(),  # ISO 639-1 in lowercase
                    'code_with_country': lang.lower(),  # Full language code in lowercase
                    'quality': quality,
                })
            except (ValueError, IndexError):
                # Skip improperly formatted entries
                continue

        # Sort by quality in descending order
        result.sort(key=lambda x: x['quality'], reverse=True)
        return result
