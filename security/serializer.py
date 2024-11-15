from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

class Serializer:
    def __init__(self):
        self.serializer = URLSafeTimedSerializer("4XPuEHT4pw6I6Bo2nQl5SnAm")

    def generate_token(self, token, salt='EAs04WE5s53HlmHqic8BJPco'):
        return self.serializer.dumps(token, salt=salt)


    def verify_token(self, token, salt='EAs04WE5s53HlmHqic8BJPco', max_age=86400):
        try:
            data = self.serializer.loads(token, salt=salt, max_age=max_age)
            return data
        except SignatureExpired:
            # Token is valid but expired
            return None
        except BadSignature:
            # Token is invalid
            return None
