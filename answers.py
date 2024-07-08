import json

class Context:
    def __init__(self,context,distance):
        self.context = context
        self.distance = distance

    def to_dict(self):
        return {
            'context': self.context,
            'distance': self.distance
        }

    @staticmethod
    def from_dict(data):
        return Context(data["context"],data["distance"])

    def __str__(self):
        return f"context:{self.context}\ndistance:{self.distance}\n"


class Answer:
    def __init__(self, answer_string, source, silhouette_score):
        self.answer_string = answer_string
        self.source = source
        self.silhouette_score = silhouette_score
        self.contexts = []

    def add_context(self,context):
        self.contexts.append(context)


    def __str__(self):
        res = ""
        for i,context in enumerate(self.contexts):
            res += f"{context}"
        res+= f"answer:{self.answer_string}, source:{self.source}, silhouette_score:{self.silhouette_score}"
        return res

    def to_dict(self):
        return {
            "answer_string": self.answer_string,
            "source": self.source,
            "silhouette_score": self.silhouette_score,
            "contexts": [context.to_dict() for context in self.contexts]
        }

    @staticmethod
    def from_dict(data):
        answer =  Answer(data["answer_string"], data["source"], data["silhouette_score"])

        answer.contexts = [Context.from_dict(context) for context in data["contexts"]]
        return answer


class AnswerSet:
    def __init__(self, index, correct_answer, question):
        self.answers_list = []
        self.index = index
        self.correct_answer = correct_answer
        self.question = question

    def add(self, answer):
        self.answers_list.append(answer)

    def get_answer_from_source(self, source):
        list = [answer for answer in self.answers_list if answer.source == source]
        return list

    def to_dict(self):
        return {
            "index": self.index,
            "correct_answer": self.correct_answer,
            "answers_list": [answer.to_dict() for answer in self.answers_list],
            "question": self.question
        }

    @staticmethod
    def from_dict(data):
        answer_set = AnswerSet(data["index"], data["correct_answer"],data["question"])
        answer_set.answers_list = [Answer.from_dict(ans) for ans in data["answers_list"]]
        return answer_set

    @staticmethod
    def save(all_answers, filename="answers.json"):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump([answer_set.to_dict() for answer_set in all_answers], file, indent=4)

    @staticmethod
    def get_from_file():
        with open("answers.json", "r", encoding="utf-8") as file:
            answers = json.load(file)
            print(answers)
            return [AnswerSet.from_dict(answer_set) for answer_set in answers]

    def __str__(self):
        res = ""
        for i, answer in enumerate(self.answers_list):
            res += f"answer {i} : {answer}\n"
        res += f"index:{self.index}, correct_answer:{self.correct_answer}\nquestion:{self.question}"
        return res