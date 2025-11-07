# from extensions import ma
# from models import Person, Embedding

# class EmbeddingSchema(ma.SQLAlchemyAutoSchema):
#     class Meta:
#         model = Embedding
#         load_instance = True
#         include_fk = True 

# class PersonSchema(ma.SQLAlchemyAutoSchema):
#     class Meta:
#         model = Person
#         load_instance = True
#     embeddings = ma.Nested(EmbeddingSchema, many=True)

# person_schema = PersonSchema()
# persons_schema = PersonSchema(many=True)


from extensions import ma 
from models import Person, Embedding 

class EmbeddingSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Embedding
        load_instance = True 


class PersonSchema(ma.SQLAlchemyAutoSchema):
    embeddings = ma.Nested(EmbeddingSchema, many=True) 
    class Meta:
        model = Person
        load_instance = True

person_schema = PersonSchema() 
persons_schema = PersonSchema(many=True) 