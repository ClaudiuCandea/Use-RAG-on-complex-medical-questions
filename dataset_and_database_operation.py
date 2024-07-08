from datasets import load_dataset
import pandas as pd
import weaviate.classes as wvc
from utils import icd_cardiovascular_diseases
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
import numpy as np

def import_dataset():
    dataset = load_dataset("bigbio/med_qa", split="train", trust_remote_code=True)
    df = pd.DataFrame(dataset)
    options_list = []
    for ind in df.index:
        df2 = pd.json_normalize(df['options'][ind])
        options = " "
        for ind2 in df2.index:
            options += f"{df2['key'][ind2]}:{df2['value'][ind2]}\n "
        options_list.append(options)
    df["options_string"] = options_list
    df = df[df['meta_info'] != 'step2&3']
    df = df.reset_index()
    print(df)
    return df

def create_weaviate_question_collection(client):
    client.collections.create(
        "MedQA",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_palm(project_id="sonic-falcon-419513"),
        generative_config=wvc.config.Configure.Generative.palm(project_id="sonic-falcon-419513"),
        properties=[
            wvc.config.Property(name="question", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="options", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
            wvc.config.Property(name="answer", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
            wvc.config.Property(name="chunkNumber", data_type=wvc.config.DataType.INT, skip_vectorization=True),
        ],
    )

def insert_question_db(df,client):
    chunk_size = 500
    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    chunks = [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} has {len(chunk)} rows")
    medQA = client.collections.get("MedQA")
    for i, chunk in enumerate(chunks):
        for index, row in chunk.iterrows():
            medQA.data.insert({
                "question": row['question'],
                "options": row['options_string'],
                "answer": row['answer'],
                "chunkNumber": i
            })
    return num_chunks

def filter_db_questions(client):
    diseases = icd_cardiovascular_diseases()
    diseases_lower = []
    for disease in diseases:
        diseases_lower.append(disease.lower())

    medQA = client.collections.get("MedQA")
    response = medQA.query.fetch_objects(
        filters=Filter.by_property("options").contains_any(diseases),
        limit=600
    )
    print(len(response.objects))
    questions1_uuid = []
    for o in response.objects:
        print(o.properties)
        questions1_uuid.append(o.uuid)
    questions1 = response.objects

    medQA = client.collections.get("MedQA")
    response = medQA.query.fetch_objects(
        filters=Filter.by_property("question").contains_any(diseases),
        limit=300
    )
    print(len(response.objects))
    questions2_uuid = []
    for o in response.objects:
        print(o.properties)
        questions2_uuid.append(o.uuid)
    questions2 = response.objects

    final_questions = questions1
    for i, q2 in enumerate(questions2):
        if questions2_uuid[i] not in questions1_uuid:
            final_questions.append(q2)

    # To use model
    questions = []
    model = VertexAI(model_name="text-bison")
    for q in final_questions:
        message = "You are an assistant that need to respond with yes if the provided question requires a diagnosis for a cardiovascular disease. At the questions that are not in this category respond with no.\n The question:" + \
                  q.properties['question']
        response = model.invoke(message)
        print(response)
        if response.find("Yes") != -1 or response.find("yes") != -1:
            question_obj = {
                "question": q.properties['question'],
                "options": q.properties['options'],
                "answer": q.properties['answer']
            }
            questions.append(question_obj)
        return questions

def filter_db_questions_old(client,num_chunks):
    medQA = client.collections.get("MedQA")
    questions = []
    for i in range(num_chunks):
        response = medQA.generate.near_text(
            query="cardiovascular system, blood, heart disease, blood vessels",
            filters=Filter.by_property("chunkNumber").equal(i),
            limit=200,
            single_prompt="You are an assistant that need to respond with yes if the provided question describes symptoms of a potential cardiovascular system disease or requires to say something related to the cardiovascular system. At the questions that are no in these categories respond with no.\n The question:{question}",
            return_metadata=MetadataQuery(distance=True)
        )
        for o in response.objects:
            if o.generated.find("No") == -1:
                if o.generated.find("no") == -1:
                    question_obj = {
                        "question": o.properties['question'],
                        "options": o.properties['options'],
                        "answer": o.properties['answer']
                    }
                    questions.append(question_obj)
    return questions

def extract_symptoms_and_diseases(new_questions):
    questions_with_symptoms = []
    model = VertexAI(model_name="text-bison")
    for q in new_questions:
        message = "You are a medical assistant that need to extract the information from the description of a medical case. Return the symptoms and the results of the examination in the same list, without separation between them.\n The medical case:" + \
                  q['context']
        response = model.invoke(message)
        print(q['question'])
        print(response)
        question_obj = {
            "context": q['context'],
            "question": q['question'],
            "options": q['options'],
            "answer": q['answer'],
            "type":q["type"],
            "symptoms": response
        }
        questions_with_symptoms.append(question_obj)

    questions_with_symptoms_and_diseases = []
    model = VertexAI(model_name="text-bison")
    for q in questions_with_symptoms:
        message = "You are a medical assistant that need to extract the pre-existing diseases from the description of a medical case. Return only the pre-existing diseases as a list. If there are no pre-existing diseases, then return the text :None\n The medical case:" + \
                  q['context']
        response = model.invoke(message)
        print(q['question'])
        print(response)
        question_obj = {
            "context": q['context'],
            "question": q['question'],
            "options": q['options'],
            "answer": q['answer'],
            "type": q["type"],
            "symptoms": q['symptoms'],
            "diseases": response
        }
        questions_with_symptoms_and_diseases.append(question_obj)
    return questions_with_symptoms_and_diseases

def embedding_from_question(question):
    embeddings = VertexAIEmbeddings(model_name = "textembedding-gecko@001")
    question_string = "The patient symptoms are:\n" + question['symptoms']
    question_embedding = embeddings.embed_query(question_string)
    question_embedding = np.array(question_embedding)
    question_embedding = question_embedding.reshape(1,-1)
    return question_embedding

def generate_option_symptoms(questions_with_symptoms_and_diseases):
    model = VertexAI(model_name="text-bison")
    generated_options = []
    for i, q in enumerate(questions_with_symptoms_and_diseases):
        print(f"We are at {i}")
        print(q['options'])
        options = q['options'].strip().split("\n")
        options_list = []
        for option in options:
            token_list = option.split(":")
            message = "Give a list of symptoms associated with this medical term. The generated answer will look like this:\n {Enter here the given term} has symptoms:\n-symptom 1\n-symptoms 2\n etc.\n If you could not generate an answer put a sentence in which you tell that.\n The term:" + \
                      token_list[1]
            response = model.invoke(message)
            options_list.append(response)
            print(response)
        generated_options.append(options_list)
    return generated_options