from spacy import displacy
import pandas as pd

NER_dict = {'O': 'O',
            'I-problem': 'problem',
            'B-problem': 'problem',

            'I-person': 'person',
            'B-person': 'person',
            'B-pronoun': 'person',

            'B-treatment': 'treatment',
            'I-treatment': 'treatment',

            'B-test': 'test',
            'I-test': 'test'}

def add_NER_tag(df):
    df['NERtag'] = df.IOBtag.replace(NER_dict)
    return df

def get_offset(lst):
    total = 0
    start_chars = []
    end_chars = []
    for token in lst:
        start = total
        end = total + len(token[0])
        total += 1 + len(token[0])
        start_chars.append(start)
        end_chars.append(end)
    return start_chars, end_chars

def make_ent(df):
    ent = 0
    ents = []
    for i in range(len(df)):
        if i > 0:
            current = df.loc[i, 'NERtag']
            prev = df.loc[i - 1, 'NERtag']
            if current != prev:
                ent += 1
        ents.append(ent)
    df['ents'] = ents
    return df

def collapse(df):
    tokens = df.groupby('ents')['token'].apply(list)
    NERtags = df.groupby('ents')['NERtag'].apply(set).apply(list).apply(lambda x: "".join(map(str, x)))
    starts = df.groupby('ents')['start'].apply(min)
    ends = df.groupby('ents')['end'].apply(max)
    prob = df.groupby('ents')['prob'].mean()
    ent_df = pd.concat([tokens, NERtags, starts, ends, prob], axis=1)
    return ent_df

def displacy_format(ent_df):
    ent_lst = []
    for i in range(len(ent_df)):
        ent_dict = {}
        if ent_df.loc[i, 'NERtag'] != 'O':
            itag = ent_df.loc[i, 'NERtag'].upper()
            istart = (ent_df.loc[i, 'start'])
            iend = (ent_df.loc[i, 'end'])
            #         print((itag, istart, iend))
            ent_dict['start'] = istart
            ent_dict['end'] = iend
            ent_dict['label'] = itag
            ent_lst.append(ent_dict)
    return ent_lst

colors = {
    "PROBLEM": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "TEST": "linear-gradient(90deg, #ffafbd, #ffc3a0)",
    "TREATMENT": "linear-gradient(90deg, #02aab0, #00cdac)",
    "PERSON": "linear-gradient(90deg, #6593f5, #73C2FB)"
}

options = {"ents": ["PROBLEM", "TEST", "TREATMENT", "PERSON"], "colors": colors}