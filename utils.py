import spacy
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def augment_pairs_df(df):
    """
    Augments a dataframe with some instances of non-related pairs of items for future model fine-tuning
    :param df: dataframe with pais of related items
    :return: df augmented with pairs of unrelated items
    """
    df['label'] = 1
    n = df.shape[0]

    # lowercase terms, replace '.' by space
    df.Term1 = df.Term1.str.lower().str.replace('.', ' ', regex=False)
    df.Term2 = df.Term2.str.lower().str.replace('.', ' ', regex=False)

    all_terms = sorted(set(df.Term1.tolist() + df.Term2.tolist()))

    related_terms_dict = defaultdict(set)
    for term1, term2 in zip(df.Term1, df.Term2):
        related_terms_dict[term1].add(term2)
        related_terms_dict[term2].add(term1)

    count_new_pairs = 0
    new_pairs = []
    while count_new_pairs < n:
        new_term_1 = random.choice(all_terms)
        new_term_2 = random.choice(list(set(all_terms) - related_terms_dict[new_term_1]))
        new_pairs.append((new_term_1, new_term_2))
        count_new_pairs += 1
    new_df = pd.DataFrame(new_pairs, columns=['Term1', 'Term2'])
    new_df['label'] = 0

    augmented_df = pd.concat([df, new_df]).sample(frac=1)

    return augmented_df


def get_word_embeddings(filepath):
  word_embeddings = {}
  with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
      word, *vector = line.split()
      word_embeddings[word] = np.array(
          vector, dtype=np.float32)

    return word_embeddings


def get_bert_vector(text, model, tokeniser):
    encoded_input = tokeniser(text, return_tensors='pt')
    output = model(**encoded_input)
    return output['pooler_output'].detach().numpy()


def get_bert_vector_dataset(terms, model, tokeniser):
    word_embeddings = {}
    for term in set(terms):
        output = get_bert_vector(term, model, tokeniser)
        word_embeddings[term] = output
    return word_embeddings


def get_avg_similarity(list1, list2, embeddings, embedding_type = 'bert'):
    similarities = []
    for term1, term2 in zip(list1, list2):
        if embedding_type == 'bert':
            vec1 = embeddings[term1]
            vec2 = embeddings[term2]
            similarities.append(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)))
        else:
            phrase1 = term1.split()
            phrase2 = term2.split()
            if any(word in embeddings for word in phrase1) and any(word in embeddings for word in phrase2):
                vec1 = sum([embeddings[word] for word in phrase1 if word in embeddings]) / len(
                    phrase1)
                vec2 = sum([embeddings[word] for word in phrase2 if word in embeddings]) / len(
                    phrase2)
                similarities.append(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)))
    return sum(similarities) / len(similarities)


if __name__ == '__main__':
    data = pd.read_csv(r'C:\Users\Anna\PycharmProjects\SentimentAnalysis\biomed\MedicalConcepts.csv')
    augmented_data = augment_pairs_df(data)
    augmented_data.to_csv(r'C:\Users\Anna\PycharmProjects\SentimentAnalysis\biomed\MedicalConcepts_augmented.csv',
                          index=False)



