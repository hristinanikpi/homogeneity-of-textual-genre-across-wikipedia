import pandas as pd
import numpy as np
import mwparserfromhell as mwp
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools
import tensorflow as tf 
from IPython.core.display import display, HTML
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from gensim.matutils import corpus2dense
from gensim.corpora import Dictionary
from gensim import models
import gensim.models.doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import scipy
import scipy.stats
import spacy
import spacy.cli
spacy.cli.download("fr_core_news_sm")
import sys 
sys.setrecursionlimit(10000)
import pywikibot as pwb

nlp = spacy.load("fr_core_news_sm")
site = pwb.Site('fr', 'wikipedia') 

def strip_node(node):
	if type(node) == mwp.nodes.text.Text:
		# to treat cases such as L'''Institute for Health Metrics and Evaluation''
		return node.value.replace("L'",'L_').replace("l'",'l_').replace("'''","").replace("''","").replace("L_","L'").replace("l_","l'") 
	
	if type(node) == mwp.nodes.template.Template:
		if node.name in ['date', 'Date-', 'date-']:
			return ' '.join([p.value.strip_code() for p in node.params])
		elif node.name in ['colonnes', 'Colonnes-']:
			return ' '.join([p.value.strip_code() for p in node.params])
		else:
			return None
  		
	if type(node) == mwp.nodes.tag.Tag:
		if node.tag.strip_code()=='b':
			return node.contents.strip_code()
		else:
			return None
  
	if type(node) == mwp.nodes.wikilink.Wikilink:
		if node.title.startswith('Catégorie:'):
			return None
		elif node.title.startswith('Fichier:'):
			return None
		elif node.title.startswith('File:'):
			return None
		elif node.title.startswith('Image:'):
			return None
		else:
			#print(node)
			return node.text.strip_code() if node.text else node.title.strip_code()
  
	if type(node) == mwp.nodes.external_link.ExternalLink and node.title != None:
		return node.title.strip_code() 

	if type(node) == mwp.nodes.heading.Heading and node.title != None:
		return node.title.strip_code()
    
	return node

def extract_text(parsedText):
  text = ''
  for i, node in enumerate(parsedText.nodes):
    node_s = strip_node(node)
    if node_s:
        text += str(node_s)
  return text.strip()

def tagged_documents(list_of_list_of_words):
	for i, list_of_words in enumerate(list_of_list_of_words):
		yield TaggedDocument(list_of_words, [i])

def save_pages_1(list_of_pages):
	for title in list_of_pages:
		page = pwb.Page(site, title)
		pickle.dump(page, open('data/l1_pages/' + title + '.pkl', 'wb'))
		
def extract_pages_1(list_of_pages):
	df = pd.DataFrame()
	G = nx.Graph()
	G.add_nodes_from(list_of_pages)
	titles = []
	texts = []
	nodes = []
	nodes_with_words = []
	words = []
	parts_of_speech = []

	for i in range(len(list_of_pages)):
		titles.append(list_of_pages[i])
		page = pickle.load(open('data/l1_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		linked_pages = page.linkedPages()
		for linked_page in linked_pages:
			if linked_page.title() in list_of_pages:
				G.add_edge(page.title(), linked_page.title())
				
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		nodes_with_words_tmp = []
		nodes_tmp = []
		for j in range(len(parsed_text.nodes)):
			node = type(parsed_text.nodes[j]) 
			if str(node).split('.')[-1][:-2] == 'Template':
				nodes_tmp.append(str(node.name))
			else:
				nodes_tmp.append(str(node).split('.')[-1][:-2])
	
			if node == mwp.nodes.text.Text:
				tmp = str(strip_node(parsed_text.nodes[j])).strip()
				doc_tmp = nlp(tmp)
				lemmatized_text_without_sw_tmp = [w.lemma_ for w in doc_tmp if w.is_stop == False]
				parts_of_speech_tmp = [w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw_tmp))]
				nodes_with_words_tmp = nodes_with_words_tmp + parts_of_speech_tmp
			else:
				if str(node).split('.')[-1][:-2] == 'Template':
					nodes_with_words_tmp.append(str(node.name))
				else:
					nodes_with_words_tmp.append(str(node).split('.')[-1][:-2])
		nodes.append(nodes_tmp)
		nodes_with_words.append(nodes_with_words_tmp)
					
		text = extract_text(parsed_text)
		texts.append(text)
		doc = nlp(text)
		lemmatized_text_without_sw = [w.lemma_ for w in doc if w.is_stop == False]
		words.append(lemmatized_text_without_sw)
		parts_of_speech.append([w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw))])

	df['Title'] = titles
	df['Text'] = texts
	df['Nodes'] = nodes
	df['Words'] = words
	df['Nodes with words'] = nodes_with_words
	df['Parts of speech'] = parts_of_speech
  
	for node in list(G.nodes):
		if G.degree[node] == 0:
			G.remove_node(node)
			df.drop(df[df['Title'] == node].index, inplace=True)
      
	df.reset_index(drop=True, inplace=True)

	return df, G	
	
def extract_pages_wsw_1(list_of_pages):
	df = pd.DataFrame()
	G = nx.Graph()
	G.add_nodes_from(list_of_pages)
	titles = []
	texts = []
	nodes = []
	nodes_with_words = []
	words = []
	parts_of_speech = []

	for i in range(len(list_of_pages)):
		titles.append(list_of_pages[i])
		page = pickle.load(open('data/l1_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		linked_pages = page.linkedPages()
		for linked_page in linked_pages:
			if linked_page.title() in list_of_pages:
				G.add_edge(page.title(), linked_page.title())
				
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		nodes_with_words_tmp = []
		nodes_tmp = []
		for j in range(len(parsed_text.nodes)):
			node = type(parsed_text.nodes[j]) 
			if str(node).split('.')[-1][:-2] == 'Template':
				nodes_tmp.append(str(node.name))
			else:
				nodes_tmp.append(str(node).split('.')[-1][:-2])
	
			if node == mwp.nodes.text.Text:
				tmp = str(strip_node(parsed_text.nodes[j])).strip()
				doc_tmp = nlp(tmp)
				lemmatized_text_tmp = [w.lemma_ for w in doc_tmp]
				parts_of_speech_tmp = ['-'.join([w.pos_, w.dep_]) for w in nlp(' '.join(lemmatized_text_tmp))]
				nodes_with_words_tmp = nodes_with_words_tmp + parts_of_speech_tmp
			else:
				if str(node).split('.')[-1][:-2] == 'Template':
					nodes_with_words_tmp.append(str(node.name))
				else:
					nodes_with_words_tmp.append(str(node).split('.')[-1][:-2])
		nodes.append(nodes_tmp)
		nodes_with_words.append(nodes_with_words_tmp)
					
		text = extract_text(parsed_text)
		texts.append(text)
		doc = nlp(text)
		lemmatized_text = [w.lemma_ for w in doc]
		words.append(lemmatized_text)
		parts_of_speech.append(['-'.join([w.pos_, w.dep_]) for w in nlp(' '.join(lemmatized_text))])

	df['Title'] = titles
	df['Text'] = texts
	df['Nodes'] = nodes
	df['Words'] = words
	df['Nodes with words'] = nodes_with_words
	df['Parts of speech'] = parts_of_speech
  
	for node in list(G.nodes):
		if G.degree[node] == 0:
			G.remove_node(node)
			df.drop(df[df['Title'] == node].index, inplace=True)
      
	df.reset_index(drop=True, inplace=True)

	return df, G	

def extract_titles_2():
	vital_page_2 = pwb.Page(site, 'Wikipédia:Articles vitaux/Niveau 2')
	parsed_vital_page_2 = mwp.parse(vital_page_2.text, skip_style_tags=True)
	text_2 = extract_text(parsed_vital_page_2)
	index_2 = text_2.index('Total actuel (98 articles)')
	pages_2 = text_2[index_2 + len('Total actuel (98 articles)'):].strip().split('\n')

	list_of_pages_2 = []
	list_of_categories_2 = []
	for i in pages_2:
		if i.find('articles') > 0:
			index = i.index('(')
			category = i[:index].strip()
		if i != '' and i[0] == '3' and i.find('\'\'\'') >= 0:
			i = i.strip()
			index = i.index('\'\'\'')
			i = i[index+len('\'\'\''):-3]
			list_of_pages_2.append(i)
			list_of_categories_2.append(category)
		elif i != '' and i[0] == '3' and i.find('\'') == -1:
			i = i.strip()
			index = i.index('   ')
			i = i[index+len('   '):]
			list_of_pages_2.append(i)
			list_of_categories_2.append(category)
		elif i != '' and i[0] != '3' and i.find('\'\'\'') >= 0:
			i = i.strip()
			index = i.index('\'\'\'')
			i = i[index+len('\'\'\''):-3]
			list_of_pages_2.append(i)
			list_of_categories_2.append(category)
		elif i != '' and i.find('(') == -1:
			list_of_pages_2.append(i.strip())
			list_of_categories_2.append(category)
	
	return list_of_pages_2, list_of_categories_2

def save_pages_2(list_of_pages):
	for title in list_of_pages:
		page = pwb.Page(site, title)
		pickle.dump(page, open('data/l2_pages/' + title + '.pkl', 'wb'))

def extract_pages_2(list_of_pages, list_of_categories):
	df = pd.DataFrame()
	G = nx.Graph()
	G.add_nodes_from(list_of_pages)
	titles = []
	texts = []
	nodes = []
	nodes_with_words = []
	words = []
	parts_of_speech = []
	for i in range(len(list_of_pages)):
		titles.append(list_of_pages[i])
		page = pickle.load(open('data/l2_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		linked_pages = page.linkedPages()
		for linked_page in linked_pages:
			if linked_page.title() in list_of_pages:
				G.add_edge(page.title(), linked_page.title())
		
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		nodes_tmp = []
		nodes_with_words_tmp = []
		for j in range(len(parsed_text.nodes)):
			node = type(parsed_text.nodes[j]) 
			if str(node).split('.')[-1][:-2] == 'Template':
				nodes_tmp.append(str(node.name))
			else:
				nodes_tmp.append(str(node).split('.')[-1][:-2])
	
			if node == mwp.nodes.text.Text:
				tmp = str(strip_node(parsed_text.nodes[j])).strip()
				doc_tmp = nlp(tmp)
				lemmatized_text_without_sw_tmp = [w.lemma_ for w in doc_tmp if w.is_stop == False]
				parts_of_speech_tmp = [w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw_tmp))]
				nodes_with_words_tmp = nodes_with_words_tmp + parts_of_speech_tmp
			else:
				if str(node).split('.')[-1][:-2] == 'Template':
					nodes_with_words_tmp.append(str(node.name))
				else:
					nodes_with_words_tmp.append(str(node).split('.')[-1][:-2])
		nodes.append(nodes_tmp)
		nodes_with_words.append(nodes_with_words_tmp)
    
		text = extract_text(parsed_text)
		texts.append(text)
		doc = nlp(text)
		lemmatized_text_without_sw = [w.lemma_ for w in doc if w.is_stop == False]
		words.append(lemmatized_text_without_sw)
		parts_of_speech.append([w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw))])

	df['Title'] = titles
	df['Category'] = list_of_categories
	df['Text'] = texts
	df['Nodes'] = nodes
	df['Words'] = words
	df['Nodes with words'] = nodes_with_words
	df['Parts of speech'] = parts_of_speech

	for node in list(G.nodes):
		if G.degree[node] == 0:
			G.remove_node(node)
			df.drop(df[df['Title'] == node].index, inplace=True)

	df.reset_index(drop=True, inplace=True)

	return df, G
	
def extract_pages_wsw_2(list_of_pages, list_of_categories):
	df = pd.DataFrame()
	G = nx.Graph()
	G.add_nodes_from(list_of_pages)
	titles = []
	texts = []
	nodes = []
	nodes_with_words = []
	words = []
	parts_of_speech = []
	for i in range(len(list_of_pages)):
		titles.append(list_of_pages[i])
		page = pickle.load(open('data/l2_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		linked_pages = page.linkedPages()
		for linked_page in linked_pages:
			if linked_page.title() in list_of_pages:
				G.add_edge(page.title(), linked_page.title())
		
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		nodes_tmp = []
		nodes_with_words_tmp = []
		for j in range(len(parsed_text.nodes)):
			node = type(parsed_text.nodes[j]) 
			if str(node).split('.')[-1][:-2] == 'Template':
				nodes_tmp.append(str(node.name))
			else:
				nodes_tmp.append(str(node).split('.')[-1][:-2])
	
			if node == mwp.nodes.text.Text:
				tmp = str(strip_node(parsed_text.nodes[j])).strip()
				doc_tmp = nlp(tmp)
				lemmatized_text_tmp = [w.lemma_ for w in doc_tmp] 
				parts_of_speech_tmp = ['-'.join([w.pos_, w.dep_]) for w in nlp(' '.join(lemmatized_text_tmp))]
				nodes_with_words_tmp = nodes_with_words_tmp + parts_of_speech_tmp
			else:
				if str(node).split('.')[-1][:-2] == 'Template':
					nodes_with_words_tmp.append(str(node.name))
				else:
					nodes_with_words_tmp.append(str(node).split('.')[-1][:-2])
		nodes.append(nodes_tmp)
		nodes_with_words.append(nodes_with_words_tmp)
    
		text = extract_text(parsed_text)
		texts.append(text)
		doc = nlp(text)
		lemmatized_text = [w.lemma_ for w in doc]
		words.append(lemmatized_text)
		parts_of_speech.append(['-'.join([w.pos_, w.dep_]) for w in nlp(' '.join(lemmatized_text))])

	df['Title'] = titles
	df['Category'] = list_of_categories
	df['Text'] = texts
	df['Nodes'] = nodes
	df['Words'] = words
	df['Nodes with words'] = nodes_with_words
	df['Parts of speech'] = parts_of_speech

	for node in list(G.nodes):
		if G.degree[node] == 0:
			G.remove_node(node)
			df.drop(df[df['Title'] == node].index, inplace=True)

	df.reset_index(drop=True, inplace=True)

	return df, G
	
def evaluate_doc2vec_2(df, corpus, tagged_corpus, params, top):
	scores = []
	for param in params: 
		model = None
		try:
			model = Doc2Vec(tagged_corpus,
                      dm = param['dm'], 
                      vector_size = param['vector_size'], 
                      window = param['window'], 
                      min_count = 1, 
                      epochs = param['epochs'], 
                      hs = param['hs'],
					  workers = 15)
			for i in range(len(corpus)):
				new_doc = corpus[i]
				test_doc_vector = model.infer_vector(new_doc)
				sims = model.dv.most_similar(positive=[test_doc_vector])
				top_sims = sims[:top]
				for j in range(len(top_sims)):
					if df['Category'][i] == df['Category'][top_sims[j][0]]:
						param['score'] = param['score'] + (top - j)
			scores.append(param)
		except Exception as error:
			print(f'Cannot evaluate model with parameters {param} because of error: {error}')
			continue
		
	return scores

def extract_titles_3():
	vital_page_3 = pwb.Page(site, 'Wikipédia:Articles vitaux')
	parsed_vital_page_3 = mwp.parse(vital_page_3.text, skip_style_tags=True)
	text_3 = extract_text(parsed_vital_page_3)
	index_3 = text_3.index('Personnalités (130 articles)')
	pages_3 = text_3[index_3:].strip().split('\n')

	list_of_pages_3 = []
	list_of_categories_3 = []
	list_of_subcategories_3 = []
	for i in pages_3:
		if len(i) > 2:
			if i.find('(') > 0:
				index = i.index('(')
				category = i[:index].strip()
			elif i != '' and i[0] == ' ' and i[1] != ' ':
				subcategory = i.strip()
			
			if i[0] == ' ' and i[1] == ' ' and i[2] != ' ' and i[2] != '\t' and i[2] != '\n':
				list_of_pages_3.append(i.strip())
				list_of_categories_3.append(category)
				list_of_subcategories_3.append(subcategory)
	return list_of_pages_3, list_of_categories_3, list_of_subcategories_3
	
def save_pages_3(list_of_pages):
	for title in list_of_pages:
		page = pwb.Page(site, title)
		pickle.dump(page, open('data/l3_pages/' + title + '.pkl', 'wb'))

def extract_pages_3(list_of_pages, list_of_categories, list_of_subcategories, start_index, end_index):
	df = pd.DataFrame()
	titles = []
	categories = []
	subcategories = []
	nodes = []
	nodes_with_words = []
	words = []
	parts_of_speech = []
	for i in range(start_index, end_index):
		titles.append(list_of_pages[i])
		categories.append(list_of_categories[i])
		subcategories.append(list_of_subcategories[i])
		page = pickle.load(open('data/l3_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		nodes_tmp = []
		nodes_with_words_tmp = []
		for j in range(len(parsed_text.nodes)):
			node = type(parsed_text.nodes[j]) 
			if str(node).split('.')[-1][:-2] == 'Template':
				nodes_tmp.append(str(node.name))
			else:
				nodes_tmp.append(str(node).split('.')[-1][:-2])
	
			if node == mwp.nodes.text.Text:
				tmp = str(strip_node(parsed_text.nodes[j])).strip()
				doc_tmp = nlp(tmp)
				lemmatized_text_without_sw_tmp = [w.lemma_ for w in doc_tmp if w.is_stop == False]
				parts_of_speech_tmp = [w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw_tmp))]
				nodes_with_words_tmp = nodes_with_words_tmp + parts_of_speech_tmp
			else:
				if str(node).split('.')[-1][:-2] == 'Template':
					nodes_with_words_tmp.append(str(node.name))
				else:
					nodes_with_words_tmp.append(str(node).split('.')[-1][:-2])
		nodes.append(nodes_tmp)
		nodes_with_words.append(nodes_with_words_tmp)
		
		text = extract_text(parsed_text)
		doc = nlp(text)
		lemmatized_text_without_sw = [w.lemma_ for w in doc if w.is_stop == False]
		words.append(lemmatized_text_without_sw)
		parts_of_speech.append([w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw))])

	df['Title'] = titles
	df['Category'] = categories
	df['Subcategory'] = subcategories
	df['Nodes'] = nodes
	df['Words'] = words
	df['Nodes with words'] = nodes_with_words
	df['Parts of speech'] = parts_of_speech

	return df

def extract_pages_wsw_3(list_of_pages, list_of_categories, list_of_subcategories, start_index, end_index):
	df = pd.DataFrame()
	titles = []
	categories = []
	subcategories = []
	nodes = []
	nodes_with_words = []
	words = []
	parts_of_speech = []
	for i in range(start_index, end_index):
		titles.append(list_of_pages[i])
		categories.append(list_of_categories[i])
		subcategories.append(list_of_subcategories[i])
		page = pickle.load(open('data/l3_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		nodes_tmp = []
		nodes_with_words_tmp = []
		for j in range(len(parsed_text.nodes)):
			node = type(parsed_text.nodes[j]) 
			if str(node).split('.')[-1][:-2] == 'Template':
				nodes_tmp.append(str(node.name))
			else:
				nodes_tmp.append(str(node).split('.')[-1][:-2])
	
			if node == mwp.nodes.text.Text:
				tmp = str(strip_node(parsed_text.nodes[j])).strip()
				doc_tmp = nlp(tmp)
				lemmatized_text_tmp = [w.lemma_ for w in doc_tmp] 
				parts_of_speech_tmp = ['-'.join([w.pos_, w.dep_]) for w in nlp(' '.join(lemmatized_text_tmp))]
				nodes_with_words_tmp = nodes_with_words_tmp + parts_of_speech_tmp
			else:
				if str(node).split('.')[-1][:-2] == 'Template':
					nodes_with_words_tmp.append(str(node.name))
				else:
					nodes_with_words_tmp.append(str(node).split('.')[-1][:-2])
		nodes.append(nodes_tmp)
		nodes_with_words.append(nodes_with_words_tmp)
		
		text = extract_text(parsed_text)
		doc = nlp(text)
		lemmatized_text = [w.lemma_ for w in doc]
		words.append(lemmatized_text)
		parts_of_speech.append(['-'.join([w.pos_, w.dep_]) for w in nlp(' '.join(lemmatized_text))])

	df['Title'] = titles
	df['Category'] = categories
	df['Subcategory'] = subcategories
	df['Nodes'] = nodes
	df['Words'] = words
	df['Nodes with words'] = nodes_with_words
	df['Parts of speech'] = parts_of_speech

	return df
	
def extract_graph_3(list_of_pages):
	df = pd.DataFrame()
	G = nx.Graph()
	G.add_nodes_from(list_of_pages)
	
	for i in range(len(list_of_pages)):
		page = pickle.load(open('data/l3_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		linked_pages = page.linkedPages()
		for linked_page in linked_pages:
			if linked_page.title() in list_of_pages:
				G.add_edge(page.title(), linked_page.title())

	return G
	
def extract_nodes_3(list_of_pages, list_of_categories, list_of_subcategories):
	df = pd.DataFrame()
	titles = []
	nodes = []
	nodes_with_words = []

	for i in range(len(list_of_pages)):
		titles.append(list_of_pages[i])
		page = pickle.load(open('data/l3_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		nodes_tmp = []
		nodes_with_words_tmp = []
		for j in range(len(parsed_text.nodes)):
			node = type(parsed_text.nodes[j]) 
			if str(node).split('.')[-1][:-2] == 'Template':
				nodes_tmp.append(str(node.name))
			else:
				nodes_tmp.append(str(node).split('.')[-1][:-2])
	
			if node == mwp.nodes.text.Text:
				tmp = str(strip_node(parsed_text.nodes[j])).strip()
				doc_tmp = nlp(tmp)
				lemmatized_text_without_sw_tmp = [w.lemma_ for w in doc_tmp if w.is_stop == False]
				parts_of_speech_tmp = [w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw_tmp))]
				nodes_with_words_tmp = nodes_with_words_tmp + parts_of_speech_tmp
			else:
				if str(node).split('.')[-1][:-2] == 'Template':
					nodes_with_words_tmp.append(str(node.name))
				else:
					nodes_with_words_tmp.append(str(node).split('.')[-1][:-2])
		nodes.append(nodes_tmp)
		nodes_with_words.append(nodes_with_words_tmp)

	df['Title'] = titles
	df['Category'] = list_of_categories
	df['Subcategory'] = list_of_subcategories
	df['Nodes'] = nodes
	df['Nodes with words'] = nodes_with_words


	return df
	
def extract_words_3(list_of_pages):
	df = pd.DataFrame()
	words = []
	parts_of_speech = []
	for i in range(len(list_of_pages)):
		page = pickle.load(open('data/l3_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		text = extract_text(parsed_text)
		doc = nlp(text)
		lemmatized_text_without_sw = [w.lemma_ for w in doc if w.is_stop == False]
		words.append(lemmatized_text_without_sw)
		parts_of_speech.append([w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw))])

	df['Words'] = words
	df['Parts of speech'] = parts_of_speech

	return df

def evaluate_doc2vec_3(df, corpus, tagged_corpus, params, top):
	scores = []
	for param in params: 
		model = None
		try:
			model = Doc2Vec(tagged_corpus,
                      dm = param['dm'], 
                      vector_size = param['vector_size'], 
                      window = param['window'], 
                      min_count = 1, 
                      epochs = param['epochs'], 
                      hs = param['hs'])
			for i in range(len(corpus)):
				new_doc = corpus[i]
				test_doc_vector = model.infer_vector(new_doc)
				sims = model.dv.most_similar(positive=[test_doc_vector])
				top_sims = sims[:top]
				for j in range(len(top_sims)):
					if df['Category'][i] == df['Category'][top_sims[j][0]] and df['Subcategory'][i] == df['Subcategory'][top_sims[j][0]]:
						param['score'] = param['score'] + (top - j)
			scores.append(param)
		except Exception as error:
			print(f'Cannot evaluate model with parameters {param} because of error: {error}')
			continue
		
	return scores

def extract_titles_4():
	vital_page_4 = pwb.Page(site, 'Wikipédia:Articles vitaux/Niveau 4')
	vital_linked_pages_4 = list(vital_page_4.linkedPages())[6:17]

	titles = []
	categories = []
	pages = []
	for linked_page in vital_linked_pages_4: 
		title = str(linked_page.title)
		title = title[title.index('(')+2:-3]
		category = title.split('/')[-1].strip()
		parsed_page = mwp.parse(linked_page.text, skip_style_tags=True)
		text = extract_text(parsed_page)
		if title == 'Wikipédia:Articles vitaux/Niveau/4/Personnalités':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Comédiens (90 articles)'):].strip().split('\n')
			for page in pages:
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Histoire':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Concepts de base (3 articles)'):].strip().split('\n')
			for page in pages:
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Géographie':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Principaux concepts (36 articles)'):].strip().split('\n')
			for page in pages:
				if page.find('(') == -1 and page.strip() != '': 
					titles_tmp.append(page.strip())
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]  
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Arts et culture':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Généralités (3 articles)'):].strip().split('\n')
			for page in pages: 
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Philosophie et religion':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Philosophie (101 articles)'):].strip().split('\n')
			for page in pages: 
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Vie quotidienne':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Vêtement et mode (36 articles)'):].strip().split('\n')
			for page in pages: 
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Société et sciences sociales':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Général (8 articles)'):].strip().split('\n')
			for page in pages: 
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip())
					categories_tmp.append(category.strip()) 
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Santé et médecine':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Concepts de base (42 articles)'):].strip().split('\n')
			for page in pages: 
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Science':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Sciences physiques (1,098 articles)'):].strip().split('\n')
			for page in pages: 
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Technologie':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Concepts de base (1 article)'):].strip().split('\n')
			for page in pages: 
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
		elif title == 'Wikipédia:Articles vitaux/Niveau/4/Mathématiques':
			titles_tmp = []
			categories_tmp = []
			pages = text[text.index('Concepts de base (59 articles)'):].strip().split('\n')
			for page in pages: 
				if page.find('(') == -1 and page.strip() != '':
					titles_tmp.append(page.strip()) 
					categories_tmp.append(category.strip())
				elif page.find('(N') > 0:
					titles_tmp.append(page[:page.index('(')].strip())
					categories_tmp.append(category.strip())
			titles = titles + titles_tmp[:-4]
			categories = categories + categories_tmp[:-4]
    		
	return titles, categories

def save_pages_4(list_of_pages):
	for title in list_of_pages:
		page = pwb.Page(site, title)
		pickle.dump(page, open('data/l4_pages/' + title + '.pkl', 'wb'))
	
def extract_graph_4(list_of_pages):
	df = pd.DataFrame()
	G = nx.Graph()
	G.add_nodes_from(list_of_pages)
	
	for i in range(len(list_of_pages)):
		page = pickle.load(open('data/l4_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		linked_pages = page.linkedPages()
		for linked_page in linked_pages:
			if linked_page.title() in list_of_pages:
				G.add_edge(page.title(), linked_page.title())

	return G
	
def extract_pages_4(list_of_pages, list_of_categories, start_index, end_index):
	df = pd.DataFrame()
	titles = []
	categories = []
	nodes = []
	nodes_with_words = []
	words = []
	parts_of_speech = []
	for i in range(start_index, end_index):
		titles.append(list_of_pages[i])
		categories.append(list_of_categories[i])
		page = pickle.load(open('data/l3_pages/' + list_of_pages[i] + '.pkl', 'rb'))
		parsed_text = mwp.parse(page.text, skip_style_tags=True)
		nodes_tmp = []
		nodes_with_words_tmp = []
		for j in range(len(parsed_text.nodes)):
			node = type(parsed_text.nodes[j]) 
			if str(node).split('.')[-1][:-2] == 'Template':
				nodes_tmp.append(str(node.name))
			else:
				nodes_tmp.append(str(node).split('.')[-1][:-2])
	
			if node == mwp.nodes.text.Text:
				tmp = str(strip_node(parsed_text.nodes[j])).strip()
				doc_tmp = nlp(tmp)
				lemmatized_text_without_sw_tmp = [w.lemma_ for w in doc_tmp if w.is_stop == False]
				parts_of_speech_tmp = [w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw_tmp))]
				nodes_with_words_tmp = nodes_with_words_tmp + parts_of_speech_tmp
			else:
				if str(node).split('.')[-1][:-2] == 'Template':
					nodes_with_words_tmp.append(str(node.name))
				else:
					nodes_with_words_tmp.append(str(node).split('.')[-1][:-2])
		nodes.append(nodes_tmp)
		nodes_with_words.append(nodes_with_words_tmp)
		
		text = extract_text(parsed_text)
		doc = nlp(text)
		lemmatized_text_without_sw = [w.lemma_ for w in doc if w.is_stop == False]
		words.append(lemmatized_text_without_sw)
		parts_of_speech.append([w.pos_ for w in nlp(' '.join(lemmatized_text_without_sw))])

	df['Title'] = titles
	df['Category'] = categories
	df['Nodes'] = nodes
	df['Words'] = words
	df['Nodes with words'] = nodes_with_words
	df['Parts of speech'] = parts_of_speech

	return df
