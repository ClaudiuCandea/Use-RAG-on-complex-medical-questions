from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def calc_silhouette_score(response):
    vector_list = []
    b = []
    cosine_dist_dict = {}
    for o in response.objects:
        b.append(o.metadata.distance)
        embedding = np.array(o.vector['default'])
        embedding = embedding.reshape(1,-1)
        vector_list.append(embedding)
    for i in range(len(vector_list)):
     for j in range(i+1,len(vector_list)):
        cosine_dist_dict[f"{i} to {j}"] = 1 - cosine_similarity(vector_list[i],vector_list[j])
        cosine_dist_dict[f"{j} to {i}"] = 1 - cosine_similarity(vector_list[i],vector_list[j])
    a = {}
    for i in range(len(vector_list)):
        sum = 0
        for j in range(len(vector_list)):
            if i!=j:
                sum += cosine_dist_dict[f"{i} to {j}"]
        a[i]=sum/(len(vector_list)-1)
    silhouette_score_final = 0
    for i in range(len(vector_list)):
        if b[i] > a[i]:
            silhouette_score = (b[i]-a[i])/(b[i]*2)
        else:
            silhouette_score = (b[i] - a[i]) / a[i]

        print(f"score {i}:{silhouette_score}")
        silhouette_score_final += silhouette_score
    silhouette_score_final /= (len(vector_list))
    print(f"Total score:{silhouette_score_final}")
    return silhouette_score_final.item()

def analyze_distances(answers_sets):
    max_max_correct = 0
    max_avg_correct = 0
    max_dist_correct = 0
    max_max = 0
    max_avg = 0
    max_dist = 0
    max_min_dist_correct = 0
    max_min_dist = 0
    max_min_correct = 0
    max_min = 0
    max_silhouette_score = 0
    max_silhouette_score_correct = 0
    for i, answerSet in enumerate(answers_sets):
        print(f"Question {i}")
        for j, answer in enumerate(answerSet.answers_list):
            if max_silhouette_score < answer.silhouette_score:
                max_silhouette_score = answer.silhouette_score
            if max_silhouette_score_correct < answer.silhouette_score:
                max_silhouette_score_correct = answer.silhouette_score
            print(f"We are at answer {j}")
            max_curr = 0
            sum_curr = 0
            min_curr = 2
            for context in answer.contexts:
                if context.distance > max_curr:
                    max_curr = context.distance
                if context.distance < min_curr:
                    min_curr = context.distance
                sum_curr += context.distance
            avg_curr = sum_curr / len(answer.contexts)
            dist = max_curr - avg_curr
            max_min_dist_curr = max_curr - min_curr
            print(
                f"avg: {avg_curr}, max: {max_curr}, min:{min_curr}, max_min_dist: {max_min_dist_curr} dist_avg_max: {dist}")
            if answer.answer_string == answerSet.correct_answer:
                print("Is the correct answer")
                if max_curr > max_max_correct:
                    max_max_correct = max_curr
                if avg_curr > max_avg_correct:
                    max_avg_correct = avg_curr
                if dist > max_dist_correct:
                    max_dist_correct = dist
                if max_min_dist_curr > max_min_dist_correct:
                    max_min_dist_correct = max_min_dist_curr
                if min_curr > max_min_correct:
                    max_min_correct = min_curr
            if max_curr > max_max:
                max_max = max_curr
            if avg_curr > max_avg:
                max_avg = avg_curr
            if dist > max_dist:
                max_dist = dist
            if max_min_dist_curr > max_min_dist:
                max_min_dist = max_min_dist_curr
            if min_curr > max_min:
                max_min = min_curr
        print("\n")
    print(
        f"max_avg: {max_avg}, max_max: {max_max},max_min:{max_min}, max_min_dist: {max_min_dist} max_dist: {max_dist}")
    print(
        f"max_avg_correct: {max_avg_correct}, max_max_correct: {max_max_correct}, max_min_correct:{max_min_correct}, max_min_dist_correct: {max_min_dist_correct} max_dist_correct: {max_dist_correct}")
    print(f"max_silhouette_score: {max_silhouette_score}, max_silhouette_score_correct: {max_silhouette_score_correct}")

