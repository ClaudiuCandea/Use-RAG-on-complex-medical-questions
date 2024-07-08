from metrics import calc_silhouette_score
from answers import Answer,Context,AnswerSet
from utils import extract_answer
from weaviate.classes.query import MetadataQuery
from langchain_google_vertexai import VertexAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def question_and_context_combined(article,question,query,limit):
    prompt  = "You are a medical assistant. You get a medical case that describes health problems and you need to make a diagnosis based on provided contexts. Give the answer in a json format with 3 fields: the selected option letter as letter, the selected option text as text and the reason for the selected option as reason.\n The case and question:" +question['question'] + "\nThe answer options are:\n" + question['options']
    print(prompt)
    response = article.generate.near_text(
        query=query,
        grouped_task=prompt,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
        include_vector=True
    )
    print(response.generated)
    silhouette_score = calc_silhouette_score(response)
    new_answer = Answer(extract_answer(response.generated), "Question and context combined", silhouette_score)
    for o in response.objects:
        print(o.properties['body'])
        print(o.metadata.distance)
        context = Context(o.properties['body'], o.metadata.distance)
        new_answer.add_context(context)
    return new_answer, response.generated

def context_and_question_separated(article,question,query,limit):
    prompt = "You are a medical assistant. You will receive a medical case and you need to answer the question based on the case description and the relevant contexts provided. Give the answer in a json format with 3 fields: the selected option letter as letter, the selected option text as text and the reason for the selected option as reason. Make sure that you close the { in the json format\n The case:" +question['context'] + "\nThe question:" + question['question'] + "\nThe answer options are:\n" +question['options']
    print(prompt)
    response = article.generate.near_text(
        query=query,
        grouped_task=prompt,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
        include_vector=True
    )
    silhouette_score = calc_silhouette_score(response)
    print(response.generated)
    new_answer = Answer(extract_answer(response.generated), "Question and context separated", silhouette_score)
    for o in response.objects:
        print(o.properties['body'])
        print(o.metadata.distance)
        context = Context(o.properties['body'], o.metadata.distance)
        new_answer.add_context(context)
    return new_answer, response.generated

def symptoms_and_diseases_separated(article,question,query,limit):
    prompt = "You are a medical assistant. You will receive a medical case and you need to answer the question based on the case description and the relevant contexts provided. Give the answer in a json format with 3 fields: the selected option letter as letter, the selected option text as text and the reason for the selected option as reason. Make sure that you close the {.\n The case:" +question['context'] + "\nThe symptoms:" + question['symptoms'] + "\nThe pre-existing diseases:" +question['diseases'] + "\nThe question:" + question['question'] + "\nThe answer options are:\n" +question['options']
    print(prompt)
    response = article.generate.near_text(
        query=query,
        grouped_task=prompt,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
        include_vector=True
    )
    print(response.generated)
    silhouette_score = calc_silhouette_score(response)
    new_answer = Answer(extract_answer(response.generated), "Symptoms and diseases extracted", silhouette_score)
    for o in response.objects:
        print(o.properties['body'])
        print(o.metadata.distance)
        context = Context(o.properties['body'], o.metadata.distance)
        new_answer.add_context(context)
    return new_answer, response.generated

def get_cases(client,query,collection_name):
    article = client.collections.get(collection_name)
    similar_cases = []
    response = article.query.near_text(
        query=query,
        limit=3,
        distance=0.20,
        return_metadata=MetadataQuery(distance=True)
    )

    for o in response.objects:
        print(o.properties['body'])
        similar_cases.append(o)
        print(o.metadata.distance)
    return similar_cases

def cases_and_info(article,similar_cases,question,query, limit):
    new_contexts = []
    cases = ""
    for context in similar_cases:
        new_contexts.append(context.properties['body'])
    if len(similar_cases) != 0:
        for case in new_contexts:
            cases += f"\n{case}"
    else:
        cases = "There are no relevant cases"
    prompt = "You are a medical assistant. You will receive a medical case and you need to answer the question based on the case description and the relevant contexts provided. Give the answer in a json format with 3 fields: the selected option letter as letter, the selected option text as text and the reason for the selected option as reason. Make sure that you close the {.\n The case:" +question['context'] + "\nThe symptoms:" + question['symptoms'] + "\nThe pre-existing diseases:" +question['diseases'] + "\nThe question:" + question['question'] + "\nThe answer options are:\n" +question['options'] + "\nSome relevant similar cases:\n" + cases
    print(prompt)
    response = article.generate.near_text(
        query=query,
        grouped_task=prompt,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
        include_vector=True
    )
    silhouette_score = calc_silhouette_score(response)
    print(response.generated)
    new_answer = Answer(extract_answer(response.generated), "Cases and diseases descriptions as contexts",
                        silhouette_score)
    for case in similar_cases:
        context = Context(case.properties['body'], case.metadata.distance)
        new_answer.add_context(context)
    for o in response.objects:
        print(o.properties['body'])
        print(o.metadata.distance)
        context = Context(o.properties['body'], o.metadata.distance)
        new_answer.add_context(context)
    return new_answer, response.generated

def use_generated_text(similar_cases, question,article,query, limit):
    new_contexts = []
    cases = ""
    for context in similar_cases:
        new_contexts.append(context.properties['body'])
    if len(similar_cases) != 0:
        for case in new_contexts:
            cases += f"\n{case}"
    else:
        cases = "There are no relevant cases"
    prompt = "You are a medical assistant. You will receive a medical case and you need to answer the question based on the case description and the relevant contexts provided. Give the answer in a json format with 3 fields: the selected option letter as letter, the selected option text as text and the reason for the selected option as reason. The text field will contain only the text from the option before the\":\". Make sure that you close the {.\n The case:" +question['context'] + "\nThe symptoms:" + question['symptoms'] + "\nThe pre-existing diseases:" +question['diseases'] + "\nThe question:" + question['question'] + "\nThe answer options are:\n" +question['options'] + "\nSome relevant similar cases:\n" + cases
    print(prompt)
    response = article.generate.near_text(
        query=query,
        grouped_task=prompt,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
        include_vector=True
    )
    silhouette_score = calc_silhouette_score(response)
    print(response.generated)
    new_answer = Answer(extract_answer(response.generated), "Generated text added", silhouette_score)
    for case in similar_cases:
        context = Context(case.properties['body'], case.metadata.distance)
        new_answer.add_context(context)
    for o in response.objects:
        print(o.properties['body'])
        print(o.metadata.distance)
        context = Context(o.properties['body'], o.metadata.distance)
        new_answer.add_context(context)
    return new_answer, response.generated

def ensemble1(answer_set,correct_answer):
    answers_list = [answer.answer_string for answer in answer_set.answers_list]
    s = set(answers_list)
    answer_fq = dict()
    for elem in s:
        count = answers_list.count(elem)
        print(f"{elem}:{count}")
        answer_fq[elem] = count
    sorted_keys = sorted(answer_fq,key=answer_fq.get, reverse=True)
    if sorted_keys[0] == correct_answer:
            return True
    return False

def ensemble2(answer_set,correct_answer):
    answers_list = [answer.answer_string for answer in answer_set.answers_list]
    s = set(answers_list)
    answer_fq = dict()
    for elem in s:
        count = answers_list.count(elem)
        print(f"{elem}:{count}")
        answer_fq[elem] = count
    sorted_keys = sorted(answer_fq,key=answer_fq.get, reverse=True)
    for elem in sorted_keys[:2]:
        if elem == correct_answer:
            return True
    return False
