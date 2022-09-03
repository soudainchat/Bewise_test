import pandas as pd
from transformers import pipeline
import warnings
import NER
import nltk
import pymorphy2
#nltk.download('punkt')

warnings.filterwarnings("ignore") 

data = pd.read_csv('test_data.csv')
data['text'] = [i.lower() for i in data['text']]
ner = NER.Ner_Extractor(model_checkpoint = 'xlm-roberta-large-finetuned-conll03-german')


def greeting(idx):
    greetings = ['здравствуйте','добрый день', 'добрый вечер', 'приветствую' 'день добрый','доброе утро']
    for row in data.index[data['dlg_id'] == idx]:
        if data['role'][row]=='manager' and any(greeting in data['text'][row] for greeting in greetings):
            return 'реплика {n} "{t}"'.format(n=data['line_n'][row],t=data['text'][row])


def farewell(idx):
    farewells = ['до свидания', 'всего доброго', 'всего хорошего', 'до скорого', 'счастливо']
    for row in data.index[data['dlg_id'] == idx]:
        if data['role'][row]=='manager' and any(farewell in data['text'][row] for farewell in farewells):
            return 'реплика {n} "{t}"'.format(n=data['line_n'][row],t=data['text'][row])

def names(idx):
    prob_thresh = 0.4
    morph = pymorphy2.MorphAnalyzer()
    li = ['меня', 'зовут', 'это', 'беспокоит', 'мое имя']
    for row in data.index[data['dlg_id'] == idx]:
        if data['role'][row] == 'manager':
            for word in nltk.word_tokenize(data['text'][row]):
                for p in morph.parse(word):
                    if 'Name' in p.tag and p.score >= prob_thresh:
                        text = [i for i in data['text'][row].split(" ")]
                        if any(i in text[text.index(word)-1:text.index(word)+1] for i in li):
                            return [word, data['line_n'][row], data['text'][row]]


def organization(idx):
    for row in data.index[data['dlg_id'] == idx]:
        if data['role'][row] == 'manager':
            doc = ner.get_entities(data['text'][row])
            list_of_orgs = [o[1] for o in doc if 'ORG' in o]
            if list_of_orgs:
                return list_of_orgs


def parse(data):

    for idx in data['dlg_id'].unique():
        print("Диалог {}".format(idx))
        print('\nПриветствие: {}'.format(greeting(idx)))
        print('\nПрощание: {}'.format(farewell(idx)))
        if names(idx):
            print('\nМенеджер представился в реплике {}: "{}"'.format(names(idx)[1], names(idx)[2]))
            print('\nИмя менеджера: {}'.format(names(idx)[0]))
        else: print('\nМенеджер не представился')
        print('\nНазвание компании: {}'.format(organization(idx)))
        print('\nМенеджер поздоровался и попрощался: {}'.format((greeting(idx)!=None) and (farewell(idx)!=None)))

        print("---------------------------------------------------------")



if __name__ == '__main__':
    parse(data)