# from extensions import db
# import json
# import numpy as np

# class Person(db.Model):
#     __tablename__ = 'persons'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), unique=True, nullable=False)
#     embeddings = db.relationship('Embedding', backref='person', lazy=True, cascade="all, delete-orphan")

#     def __repr__(self):
#         return f'<Person {self.name}>'

# class Embedding(db.Model):
#     __tablename__ = 'embeddings'
#     id = db.Column(db.Integer, primary_key=True)
#     person_id = db.Column(db.Integer, db.ForeignKey('persons.id'), nullable=False)
#     _embedding_data = db.Column(db.Text, nullable=False) 

#     @property
#     def embedding_data(self):
#         """Gets the embedding as a numpy array."""
#         if self._embedding_data:
#             return np.array(json.loads(self._embedding_data), dtype=np.float64)
#         return None

#     @embedding_data.setter
#     def embedding_data(self, value):
#         """Sets the embedding from a numpy array or list, storing as JSON string."""
#         if isinstance(value, np.ndarray):
#             self._embedding_data = json.dumps(value.tolist())
#         elif isinstance(value, list):
#             self._embedding_data = json.dumps(value)
#         else:
#             raise ValueError("Embedding data must be a numpy array or a list.")

#     def __repr__(self):
#         return f'<Embedding {self.id} for Person ID {self.person_id}>'

# from extensions import db # Import db instance from extensions.py
# import json
# import numpy as np

# class Face(db.Model):
#     __tablename__ = 'faces'

#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     _encoding = db.Column('encoding', db.Text, nullable=False) # Store as JSON string

#     @property
#     def encoding(self):
#         # Deserialize JSON string to numpy array
#         if self._encoding:
#             return np.array(json.loads(self._encoding))
#         return None

#     @encoding.setter
#     def encoding(self, value):
#         # Serialize numpy array or list to JSON string
#         if isinstance(value, np.ndarray):
#             self._encoding = json.dumps(value.tolist())
#         elif isinstance(value, list): # Allow setting with a list too
#              self._encoding = json.dumps(value)
#         else:
#             raise ValueError("Encoding must be a numpy array or a list.")

#     def __repr__(self):
#         return f'<Face {self.name} (ID: {self.id})>'

from extensions import db
import json
import numpy as np

class Person(db.Model):
    __tablename__ = 'persons'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False) 
    embeddings = db.relationship('Embedding', backref='person', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Person {self.name} (ID: {self.id})>'

class Embedding(db.Model):
    __tablename__ = 'embeddings'
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'), nullable=False)
    _embedding_data = db.Column('embedding_data', db.Text, nullable=False) 

    @property
    def embedding_data(self):
        if self._embedding_data:
            return np.array(json.loads(self._embedding_data), dtype=np.float64)
        return None

    @embedding_data.setter
    def embedding_data(self, value):
        if isinstance(value, np.ndarray):
            self._embedding_data = json.dumps(value.tolist())
        elif isinstance(value, list): 
             self._embedding_data = json.dumps(value)
        else:
            raise ValueError("Embedding data must be a numpy array or a list.")

    def __repr__(self):
        return f'<Embedding ID: {self.id} for Person ID: {self.person_id}>'
