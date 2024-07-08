import weaviate.classes as wvc
from chunking import word_splitter, get_chunks_fixed_size, get_chunks_fixed_size_with_overlap, get_chunks_by_paragraph, get_chunks_by_paragraph_and_min_length
import os
from langchain_google_vertexai import VertexAI


def create_collection(client,name,include_generated_text,model):
    if not include_generated_text:
        if model == "bison":
            client.collections.create(
                name,
                vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_palm(project_id="sonic-falcon-419513"),
                generative_config=wvc.config.Configure.Generative.palm(project_id="sonic-falcon-419513"),
                properties=[  # properties configuration is optional
                    wvc.config.Property(name="body", data_type=wvc.config.DataType.TEXT),
                ],
            )
        else:
            client.collections.create(
                name,
                vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_palm(project_id="sonic-falcon-419513"),
                generative_config=wvc.config.Configure.Generative.palm(project_id="sonic-falcon-419513",model_id="gemini-1.0-pro-002"),
                properties=[  # properties configuration is optional
                    wvc.config.Property(name="body", data_type=wvc.config.DataType.TEXT),
                ],
            )

    else:
        if model == "bison":
            client.collections.create(
                name,
                vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_palm(project_id="sonic-falcon-419513"),
                generative_config=wvc.config.Configure.Generative.palm(project_id="sonic-falcon-419513"),
                properties=[  # properties configuration is optional
                    wvc.config.Property(name="body", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT,skip_vectorization=True),
                ],
            )
        else:
            client.collections.create(
                name,
                vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_palm(project_id="sonic-falcon-419513"),
                generative_config=wvc.config.Configure.Generative.palm(project_id="sonic-falcon-419513",model_id="gemini-1.0-pro-002" ),
                properties=[  # properties configuration is optional
                    wvc.config.Property(name="body", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT,skip_vectorization=True),
                ],
            )

def insert_into_collection(client,name):
    article = client.collections.get(name)
    with open("./CardiovascularSystem.txt", 'r', encoding='utf-8') as file:
        file_contents = file.read()
        chunks = get_chunks_by_paragraph_and_min_length(file_contents)
        print(len(chunks))
        for chunk in chunks:
            article.data.insert(
                {"body": chunk}
            )

def insert_into_collection_generated(client,name,questions):
    article = client.collections.get(name)

    with open("./CardiovascularSystem.txt", 'r', encoding='utf-8') as file:
        file_contents = file.read()
        chunks = get_chunks_by_paragraph_and_min_length(file_contents)
        print(len(chunks))
        for chunk in chunks:
            article.data.insert(
                {"body": chunk,
                 "type": "book"}
            )
    model = VertexAI(model_name="text-bison")
    for q in questions:
        options = q['options'].strip().split("\n")
        for option in options:
            message = "Give a description for the next medical term.\n The term:" + option
            response = model.invoke(message)
            print(response)
            article.data.insert(
                {"body": response,
                 "type": "generated"}
            )

