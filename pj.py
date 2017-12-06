import pandas as pd
import string
import random 
import copy
import numpy as np
from sklearn.metrics import f1_score
import nltk
#nltk.download()

cachedStopWords = stopwords.words("english")
# ture_txt = positive reviews, false_txt = negative reviews
class nb_class:
    def data_prep(self, path_true_txt, path_false_txt):
        # read general delimited file into data frame (txt files)
        with open(path_true_txt, encoding='ISO-8859-1') as f:
            true_text = f.readlines()
        with open(path_false_txt, encoding='ISO-8859-1') as f:
            false_text = f.readlines()
        true_text = [x.strip() for x in true_text]
        false_text = [x.strip() for x in false_text]
        # print (true_text)
        # shuffle datasets and split into train, dev, test
        SEED = 120
        random.seed(SEED)
        random.shuffle(true_text)
        random.shuffle(false_text)
        training_size = int(len(true_text)*0.7)
        dev_size = int(len(true_text)*0.15)
        
        true_train = true_text[:training_size]
        true_dev = true_text[training_size:(training_size+dev_size)]
        true_test = true_text[(training_size+dev_size):]

        false_train = false_text[:training_size]
        false_dev = false_text[training_size:(training_size+dev_size)]
        false_test = false_text[(training_size+dev_size):]
        return true_train,false_train,true_dev,false_dev,true_test,false_test
    # create a dictionary of all text files
    def dict_file (self,true_text,false_text):
        count = {}
        # debug: extend
        true_text_copy = copy.deepcopy(true_text)
        true_text_copy.extend(false_text)
        for t in true_text_copy:
            for word in t.split():
                if word not in string.punctuation:
                	if word not in cachedStopWords:                    
                        if word not in count:
                            count[word] = 1
                        else:
                            count[word] += 1
        #filter out all the words with frequence less than 2
        # with open('stopwords.txt') as f:
        #     stopwords = f.readlines()
        # stopwords = [w.strip() for w in stopwords]
        # stopwords = []
        # d = dict((k, v) for k, v in count.items() if v > 1 and k not in stopwords)
        d = dict((k, v) for k, v in count.items() if v > 1)
        return d
    # 
    def count_prob(self,true_text,false_text,dictionary):
        ture_count = {}
        false_count = {}

        prior_prob = float(len(true_text))/((len(true_text)+len(false_text)))
        
        for t in true_text:
            for word in t.split():
                if word not in string.punctuation:
                    if word in dictionary:
                        if word not in ture_count:
                            ture_count[word] = 1
                        else:
                            ture_count[word] += 1

        for t in false_text:
            for word in t.split():
                if word not in string.punctuation:
                    if word in dictionary:
                        if word not in false_count:
                            false_count[word] = 1
                        else:
                            false_count[word] += 1
        true_dic_prob = {}
        false_dic_prob = {}
        for word in dictionary.keys():
            if word not in ture_count:
                word_freq = 0.
            else:
                word_freq = ture_count[word]
            true_dic_prob[word] = (word_freq + 1.)/(sum(ture_count.values())+len(dictionary))
            if word not in false_count:
                word_freq = 0.
            else:
                word_freq = false_count[word]
            false_dic_prob[word] = (word_freq + 1.)/(sum(false_count.values())+len(dictionary))
        return true_dic_prob,false_dic_prob,prior_prob

    def bayes(self,true_dic_prob,false_dic_prob,pred_text,dictionary,prior_prob):
        pred_label = []
        for pred in pred_text:
            #calculate the true prob
            true_prob = prior_prob
            false_prob = 1-prior_prob
            for word in pred.split():
                if word not in true_dic_prob:
                    true_prob_word = 1
                else:
                    true_prob_word = true_dic_prob[word]
                true_prob *= true_prob_word
                if word not in false_dic_prob:
                    false_prob_word = 1
                else:
                    false_prob_word = false_dic_prob[word]
                false_prob *= false_prob_word
            if true_prob>false_prob:
                pred_label.append(1)
            else:
                pred_label.append(0)
        return pred_label
    def getacc(self,true_pred_label,false_pred_label):
        total = len(true_pred_label)+len(false_pred_label)
        acc = 0
        for label in true_pred_label:
            if label == 1:
                acc += 1
        for label in false_pred_label:
            if label ==0:
                acc +=1


        return float(acc)/total



if __name__ == '__main__':
    #extraction of description data from csv file and split into mental==true to pos
    # mental == false to neg
    df = pd.read_csv('~/Desktop/shooting_data_with_county_covariates.csv', encoding='ISO-8859-1')
    df_text = df.ix[:,['description','mental']]
    df_pos = df_text[df_text['mental']== True]['description']
    df_neg = df_text[df_text['mental'] == False]['description']
    df_pos = np.savetxt(r'des_pos.txt',df_pos, fmt='%5s')
    df_neg = np.savetxt(r'des_neg.txt', df_neg, fmt='%5s')

    path_false_txt = './des_pos.txt'
    path_true_txt = './des_neg.txt'
    true_train,false_train,true_dev,false_dev,true_test,false_test = nb_class().data_prep(path_true_txt,path_false_txt)
    dictionary = nb_class().dict_file(true_train,false_train)
    true_dic_prob,false_dic_prob,prior_prob = nb_class().count_prob(true_train,false_train,dictionary)
    pred_true_dev = nb_class().bayes(true_dic_prob,false_dic_prob,true_dev,dictionary,prior_prob)
    pred_false_dev = nb_class().bayes(true_dic_prob,false_dic_prob,false_dev,dictionary,prior_prob)
    print ("The acc on the dev set is",nb_class().getacc(pred_true_dev,pred_false_dev))
    # test on test set







