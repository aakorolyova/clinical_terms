import re
import spacy
import random
import pandas as pd
from collections import defaultdict

nlp = spacy.load('en_core_web_sm')
def generate_vocab(df):
    tokens = [token.text.lower() for note in df.notes
              for token in nlp(note)]
    vocab = sorted(set(tokens))
    print('Total number of tokens: {0}. Number of unique tokens: {1}'.format(len(tokens), len(vocab)))
    return vocab

def save_vocab(vocab, output_path):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for token in vocab:
            output_file.write(token + '\n')

def split_sentences_text(text):
    return [sent for sent in nlp(text).sents]

def clean_text(text):
    return re.sub('([,:;)])(\\w)', '\\1 \\2', text)

def clean_clin_notes(df):
    df['notes'] = df.notes.apply(lambda x: clean_text(x))
    return df

def split_sentences_df(df):
    df['sentences'] = df.notes.apply(lambda x: split_sentences_text(x))
    new_df = pd.DataFrame([[category, sentence] for category, note, sentences in df.values
            for sentence in sentences
        ], columns=['category', 'sentence'])
    return new_df

def clean_med_concepts(df):
    """
    Cleans a dataframe: lowercase terms, replace '.' by space
    :param df: dataframe with pais of related items
    :return: cleaned dataframe
    """
    df.Term1 = df.Term1.str.lower().str.replace('.', ' ', regex=False)
    df.Term2 = df.Term2.str.lower().str.replace('.', ' ', regex=False)
    return df

def augment_pairs_df(df, vocab, num_to_add):
    """
    Augments a dataframe with some instances of non-related pairs of items for future model fine-tuning
    :param df: dataframe with pais of related items
    :param vocab: vocabulary to take unrelated terms from
    :param num_to_add: number of pairs of unrelated terms to add
    :return: df augmented with pairs of unrelated items
    """
    df['label'] = 1

    related_terms_dict = defaultdict(set)
    for term1, term2 in zip(df.Term1, df.Term2):
        related_terms_dict[term1].add(term2)
        related_terms_dict[term2].add(term1)

    count_new_pairs = 0
    new_pairs = []
    vocab_alpha = [token for token in vocab if token.isalpha()]
    while count_new_pairs < num_to_add:
        new_term_1 = random.choice(vocab_alpha)
        new_term_2 = random.choice(list(set(vocab_alpha) - related_terms_dict[new_term_1]))
        new_pairs.append((new_term_1, new_term_2))
        count_new_pairs += 1
    new_df = pd.DataFrame(new_pairs, columns=['Term1', 'Term2'])
    new_df['label'] = 0

    augmented_df = pd.concat([df, new_df]).sample(frac=1)

    return augmented_df


if __name__ == '__main__':
    # split clinical notes in sentences - for further MLM training
    df = pd.read_csv(r'data/ClinNotes.csv')
    df_cleaned = clean_clin_notes(df)
    df_sentences = split_sentences_df(df_cleaned)
    df_sentences.to_csv(r'data/ClinNotes_sentences.csv', index=False)

    # save clinical notes vocab
    vocab = generate_vocab(df_cleaned)
    save_vocab(vocab, r'data/vocab.txt')

    # clean medical concepts: lowercase, replace full stop by space
    data = pd.read_csv(r'data/MedicalConcepts.csv')
    data_cleaned = clean_med_concepts(data)
    data_cleaned.to_csv(r'data/MedicalConcepts_cleaned.csv', index=False)

    # augment the set of related terms with some unrelated pairs of terms
    n = data_cleaned.shape[0]
    augmented_data = augment_pairs_df(data_cleaned, vocab, n * 2)
    augmented_data.to_csv(r'data/MedicalConcepts_augmented.csv',
                          index=False)



