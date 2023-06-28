import func 
from collections import Counter 

df_2 = func.pickle.load(open('data/l2_data_wsw.pkl', 'rb'))
list_of_pages_2 = df_2['Title']


def extract_lexico_grammatical_features(doc):
    # word types 
    pos = Counter(['POS: ' + w.pos_ for w in doc])
    df_pos = func.pd.DataFrame(pos, index=[0])
    # dependencies
    dep = Counter(['DEP: ' + w.dep_ for w in doc])
    df_dep = func.pd.DataFrame(dep, index=[0])
    # morphologies 
    morph = Counter(['MORPH: ' + str(w.morph) for w in doc])
    df_morph = func.pd.DataFrame(morph, index=[0])
    
    df = func.pd.concat([df_pos, df_dep, df_morph], axis=1)
    
    # number of sentences 
    sents = [[token.text for token in sent] for sent in doc.sents]
    df['NUM: sentences '] = len(sents)
    # number of words
    df['NUM: words '] = len(doc)
    # number of entities 
    df['NUM: named entities '] = len(doc.ents)
    
    return df 

'''
page = func.pickle.load(open('data/l2_pages/' + list_of_pages_2[0] + '.pkl', 'rb'))
parsed_text = func.mwp.parse(page.text, skip_style_tags=True)
text = func.extract_text(parsed_text)
doc = func.nlp(text)
df = extract_lexico_grammatical_features(doc)
'''

df = func.pd.DataFrame()
for i in range(len(list_of_pages_2)):
    page = func.pickle.load(open('data/l2_pages/' + list_of_pages_2[i] + '.pkl', 'rb'))
    parsed_text = func.mwp.parse(page.text, skip_style_tags=True)
    text = func.extract_text(parsed_text)
    doc = func.nlp(text)
    df_tmp = extract_lexico_grammatical_features(doc)
    df = func.pd.concat([df, df_tmp], axis=0)
    
df.fillna(0, inplace=True)
df.reset_index(drop=True, inplace=True)


func.pickle.dump(df, open('features/l2_features.pkl', 'wb'))