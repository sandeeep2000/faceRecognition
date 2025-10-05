import os


class Config:
    """
    Configuration class for the Flask application.

    """

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', 'postgresql://sandeep:8142136367@localhost:5432/face_recognition_db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    FACE_RECOGNITION_TOLERANCE = 0.6
