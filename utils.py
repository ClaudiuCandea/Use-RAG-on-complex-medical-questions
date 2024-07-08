import simple_icd_10 as icd
import json
import re

def icd_cardiovascular_diseases():
    initial_code = "IX"
    codes = icd.get_children(initial_code)
    names = []
    while codes:
        code = codes.pop(0)
        if icd.is_leaf(code):
            names.append(icd.get_description(code))
        else:
            codes.extend(icd.get_children(code))
    return names


def extract_answer(response):
    first_pos = response.find('{')
    second_pos = response.find('}')
    json_string = response[first_pos:second_pos + 1]
    fields = json_string.split("\",")
    for field in fields:
        if field.find("text") != -1:
            strs = field.split(":")
            gen_answer = strs[1].replace("\"", "").strip()
            print(gen_answer)
            return gen_answer
    return ""


def verify_answer(response, answer):
    gen_answer = extract_answer(response)
    if gen_answer == "":
        return False
    if gen_answer == answer:
        print("Is the correct answer")
        return True
    return False

def separate_question(paragraph):
    sentences = re.split(r'(?<=[.!?])\s+|\n', paragraph)

    if len(sentences) < 2:
        raise ValueError("The paragraph does not contain multiple sentences.")

    question = sentences[-1]
    n = 2
    while question.find('?') == -1 and question.find(":") == -1:
        question = sentences[-n] + question
        n += 1
    if question[-1] == ')':
        question = sentences[-n] + question
        n += 1
    rest_of_text = ' '.join(sentences[:-n + 1])

    return question, rest_of_text

def create_new_questions(questions):
    new_questions = []
    for question in questions:
        new_question, rest_of_text = separate_question(question['question'])
        new_question_obj = {
            "question": new_question,
            "context": rest_of_text,
            "options": question['options'],
            "answer": question['answer'],
            "type": question['type']
        }
        new_questions.append(new_question_obj)
    return new_questions


