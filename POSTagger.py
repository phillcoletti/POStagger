# POSTagger
# 5/14/2014
# Phillip Coletti
# Dartmouth College CS-189 Probabilistic Graphical Models (Spring 2014)

import os
import os.path
import errno
import sys
import pandas as pd
import numpy as np
import math
import random
import timeit
import csv
import pickle
<<<<<<< HEAD
=======
import operator
>>>>>>> v2
#from nltk.corpus import brown

eps = 10e-50
NEGINF = -10e10

class POSTagger:
    def __init__(self, directionality, feature_set, support_cutoff, tag_cutoff):
        self.directionality = directionality
        self.support_cutoff = support_cutoff
        self.feature_set = feature_set
        self.initialize_log()
        if not os.path.isfile('word_features.csv'):
            #Extract features from all sentences
    #        feature_list = []
            feature_names = ['sentence_num', 't_i-2', 't_i-1', 't_i', 't_i+1', 't_i+2', 
                            'w_i-2', 'w_i-1', 'w_i', 'w_i+1', 'w_i+2', 
                            #'w_i-1,i', 'w_i,i+1', 'w_i,t_i-1', 'w_i,t_i+1', 
                            't_i-2,i-1', 't_i-1,i+1', 't_i+1,i+2'] 
            feature_writer = csv.writer(open('word_features.csv', 'wb'))
            feature_writer.writerow(feature_names)
            i = 0;
            words_file = open('brown-words.txt', 'r')
            tags_file = open('brown-tags.txt', 'r')
            for sentence in words_file:
                sentence = sentence.replace("\n", "")
                sentence = sentence.split(" ")
                tagline = tags_file.readline()
                tagline = tagline.split(" ")
                tagline.pop(-1)
                if len(sentence) >= 2:
                    w1 = sentence.pop(0)
                    w2 = sentence.pop(0)
                    t1 = tagline.pop(0)
                    t2 = tagline.pop(0)
                    tags = ['start', 'start', 'start', t1, t2]
                    words = [None, None, None, w1, w2]
    #                import pdb; pdb.set_trace()
                    sentence.append('')
                    sentence.append('')
                    tagline.append('end')
                    tagline.append('end')
                    for (w,t) in zip(sentence, tagline):
                        #Shift all words and tags down one slot
                        tags.pop(0)
                        tags.append(t)
                        words.pop(0)
                        words.append(w)
                        extra_features = [#(words[1], words[2]), (words[2], words[3]),    #lexical
                                          #(words[2], tags[1]), (words[2], tags[3]),      #lexical-word combo
                                          tags[0] + '_' + tags[1], tags[1] + '_' + tags[3], tags[3] + '_' + tags[4]]  #tag sequence
                        features = [i] + tags + words + extra_features
                        feature_writer.writerow(features)
    #                    feature_list.append(features)
    #                print i
                    i = i + 1
		#convert to DataFrame
        self.word_features = pd.DataFrame.from_csv('word_features.csv', index_col=False)
		#Calculate Tagset
        frequency = pd.crosstab(rows=self.word_features['t_i'], cols=self.word_features['t_i+1'])
        self.tag_set = {}
        for col in frequency.columns:
            tags = frequency[col][frequency[col] >= tag_cutoff].index.values
            self.tag_set[col] = tags

    def initialize_log(self):
        filenamebase = "__directionality=" + self.directionality
        filenamebase = filenamebase + "__support=%d" % self.support_cutoff
        filenamebase = filenamebase + "__features="
        for feature in self.feature_set:
            filenamebase = filenamebase + "," + feature
        self.logfilename = "log" + filenamebase + ".log"
        self.datafilename = "data" + filenamebase + ".csv"
        try:
            os.remove(self.logfilename)
        except OSError as e:
            if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                raise # re-raise exception if a different error occured

    def log(self, message):
        logfile = open(self.logfilename, "a")
        logfile.write(message + "\n")
        print message
    
    def fixedEval(self):
        data_sets = pickle.load( open( "data_sets.p", "rb" ) )
        print "Training..."
        self.train(train_sentence_nums=data_sets['train_set'])
        print "Testing..."
        self.test(test_sentence_nums=data_sets['test_set'])			

    #####################################################
    # Evaluates the model on a random set of N sentences;
    # Trains on all other sentences
    #####################################################
    def randEvalL(self, N):
        sentence_nums = self.word_features['sentence_num'].unique()
        sample = random.sample(xrange(len(sentence_nums)), N)
        test_set = sentence_nums[sample]
        removeset = set(sample)
        train_set = [v for i, v in enumerate(sentence_nums) if i not in sample]
        print "Training..."
        self.train(train_sentence_nums=train_set)
        print "Testing..."
        self.test(test_sentence_nums=test_set)

    #####################################################
    # Evaluates the model on a random set of N sentences;
    # Trains on all other sentences
    #####################################################
    def randEvalLR(self, N):
        sentence_nums = self.word_features['sentence_num'].unique()
        sample = random.sample(xrange(len(sentence_nums)), N)
        test_set = sentence_nums[sample]
        removeset = set(sample)
        train_set = [v for i, v in enumerate(sentence_nums) if i not in sample]
        print "Training..."
        self.train(train_sentence_nums=train_set)
        print "Testing..."
        self.test(test_sentence_nums=test_set)

    #####################################################
    # Evaluates the model on a random set of N sentences;
    # Trains on all other sentences
    #####################################################
    def randEvalR(self, N):
        sentence_nums = self.word_features['sentence_num'].unique()
        sample = random.sample(xrange(len(sentence_nums)), N)
        test_set = sentence_nums[sample]
        removeset = set(sample)
        train_set = [v for i, v in enumerate(sentence_nums) if i not in sample]
        print "Training..."
        self.train(train_sentence_nums=train_set)
        print "Testing..."
        self.test(test_sentence_nums=test_set)

    ####################################################################
    # Trains the prior and conditional probabilities
    # prior[t_i] = log_probability(t_i=tag_string)
    # conditional[feature_name][feature_value][t_i] = log_conditional_probability(t_i=tag_string | feature_name=feature_value)
    # If the conditioanl probability is NOT in the conditional dictioanary, it is below the support cutoff; sub in the prior
    #####################################################################
    def train(self, train_sentence_nums):
        #select subest of training data from training_sentences list only
        word_features = self.word_features[self.word_features['sentence_num'].isin(train_sentence_nums)]
        #word_features = self.word_features
        #Caluclate conditional probabilities		
        self.log_prior = None
        self.log_conditional = {}
        feature_names = list(word_features.columns.values)
        if 't_i' in feature_names:
            feature_names.remove('t_i')
        if 'sentence_num' in feature_names:
            feature_names.remove('sentence_num')
        for feature_name in feature_names:
            frequency = pd.crosstab(rows=word_features['t_i'], cols=word_features[feature_name]) #calculate frequency table for feature vs. t_i
            feature_sum = frequency.sum().astype(float)  #sum across all possible t_i's for a given feature value
            supported = feature_sum[(feature_sum >= self.support_cutoff)].index
            #		    unsupported = feature_sum[(feature_sum < 2)].index  #Support Cutoff
            p = frequency / feature_sum    #normalize to become conditional probability of t_i (given feature value)
            p = p[supported]        #ignore columns below support cutoff
            p[p < eps] = eps
            logp = np.log(p)
            #		    p[unsupported] = None  #don't count those feature values that are below cutoff
            self.log_conditional[feature_name] = logp.to_dict()

            #Calculate Prior Probabilities
            prior_count = word_features['t_i'].value_counts()
            total_count = prior_count.sum()
            prior_prob = prior_count / float(total_count)
            prior_prob[prior_prob < eps] = eps
            self.log_prior = np.log(prior_prob)

    def test(self, test_sentence_nums):
        total_correct = 0
        total_words = 0
        count = 1
        print "sentence_nums = ", test_sentence_nums
        for sentence_num in test_sentence_nums:
            sentence = self.word_features[self.word_features['sentence_num'] == sentence_num]
            #set index to begin at 0 for consistency with tag predictions
            sentence = sentence.reset_index()
            tic = timeit.default_timer() 		#start timer
            if self.directionality is "L":
                cache = self.bestScoreL(sentence)
                tags = self.getTagPredictionsL(sentence, cache)
            elif self.directionality is "LR":
                cache = self.bestScoreLR(sentence)
                tags = self.getTagPredictionsLR(sentence, cache)
            elif self.directionality is "R":
                cache = self.bestScoreR(sentence)
                tags = self.getTagPredictionsR(sentence, cache)
            elif self.directionality is "None":
                tags = self.getTagPredictionsW(sentence)
            else:
                self.log("Directionality %s invalid" % self.directionality)
                sys.exit(1)
            toc = timeit.default_timer()		#end timer
            correct = sum(sentence['t_i'] == tags)
            words = len(tags)
            if words > 0:
                avg_time = (toc - tic) / words
                accuracy = correct / float(words)
            else:
                avg_time = 0
                accuracy = 0
            total_correct += correct
            total_words += words
            self.log("Sentence %d: %f" % (sentence_num, accuracy))
            self.log("%f s avg for %d words" % (avg_time, words))
            if total_words > 0:
                total_accuracy = total_correct / float(total_words)
            else:
                total_accuracy = 0
            self.log("Accuracy on %d sentences: %f" % (count, total_accuracy) )
            if count % 20 == 0:
                tested = self.word_features.ix[self.word_features['tag_prediction'].notnull()]
                tested.to_csv(self.datafilename)
            count = count + 1
        tested = self.word_features.ix[self.word_features['tag_prediction'].notnull()]
        tested.to_csv(self.datafilename)
        self.log("Finished Testing!")

    def getConditionalProb(self, features, t_i):
        #if the conditional probability is equal to None, then we assign a prior
        conditionalProbVal = 0
        #conditional for word i
        for feature_name in self.feature_set:
            feature_value = features[feature_name]
            if (feature_value in self.log_conditional[feature_name].keys()) and (self.log_conditional[feature_name][feature_value][t_i] != None):
                conditionalProbVal = conditionalProbVal + self.log_conditional[feature_name][feature_value][t_i]
            else:
                conditionalProbVal = conditionalProbVal + self.log_prior[t_i]	

        return conditionalProbVal


    def getTagPredictionsW(self, sentence):
        import pdb; pdb.set_trace()
        tags = []
        for i in range(len(sentence)):
            probs = []
            tagset = self.word_features['t_i'].unique()
            for ti in tagset:
                probs.append(self.getConditionalProb(sentence.ix[i], ti)) #will be conditional probability of Ti given features P(ti|tim1,tip1,wi)
            index, value = max(enumerate(probs), key=operator.itemgetter(1))
            tags.append(tagset[index])
        sentence['tag_prediction'] = tags
        if 'tag_prediction' not in self.word_features.columns.values:
            print "Could NOT find tag_prediction column"
            self.word_features['tag_prediction'] = None
        for i in range(len(sentence)):
            self.word_features['tag_prediction'].ix[sentence.ix[i]['index']] = sentence['tag_prediction'].ix[i]
        return(tags)



    ########################################################
    # go through set of sentences and return tags
    ########################################################
    def getTagPredictionsLR(self, sentence, cache):
        # create the tags list and run the algorithm to get the cache
        # working from the back to the front of the sentence
        tags = []
        tip1 = 'end'
        ti = 'end'
        tim1 = 'end'
        i = len(sentence) + 1

        #loop through all indices of the cache until we reach the start
        # appends tags to list in reverse order
        while cache.has_key((i, tim1, ti, tip1)):
            (maxVal, maxTag) = cache[(i, tim1, ti, tip1)]
            tags.append(maxTag)
            tip1 = ti
            ti = tim1
            tim1 = maxTag
            i = i - 1

        # reverse order so the list is from start of sentence to finish
        # and remove the two 'start' tags at the front
        tags.reverse()
        tags.pop(0)
        tags.pop(0)
        #put the tag predictions into the main word feature_vector for analysis
#		import pdb; pdb.set_trace()
        sentence['tag_prediction'] = tags
        if 'tag_prediction' not in self.word_features.columns.values:
            print "Could NOT find tag_prediction column"
            self.word_features['tag_prediction'] = None
        for i in range(len(sentence)):
            self.word_features['tag_prediction'].ix[sentence.ix[i]['index']] = sentence['tag_prediction'].ix[i]

        return tags

    def bestScoreLR(self, sentence):
        #for each sentence best score is called, and then it makes this cache
        #the cache stores values for each of the parameters given and can then be used to determine the tokens
        cache = {}

        #let n equal the last index in the list
        n = len(sentence) - 1
        self.bestScoreSubLR(n+2, ('end','end','end'), sentence, cache)

        return cache


    def bestScoreSubLR(self, i, (tim1, ti, tip1) , sentence, cache):

        #takes a cache from best score and adds return values and tags to it
        if cache.has_key((i, tim1, ti, tip1)):
            (maxVal, maxTag) = cache[(i, tim1, ti, tip1)]
            return maxVal

#        print i, ti

        #left boundry case
        if (i == -1):
            if ((tim1, ti)==('start','start')):
                return 0
            else:
                return NEGINF

        #calculate conditional probability
        if ti == 'end':		#right boundary case
            probTiGiven = 0
        else:
            probTiGiven = self.getConditionalProb(sentence.ix[i], ti) #will be conditional probability of Ti given features P(ti|tim1,tip1,wi)

        #recursive case, loop through all tags, find the tag and val with the maximum likelihood
        maxVal = NEGINF
        maxTag = None
        #Left Boundary: Only check 'start' tag before index 0
        if i <= 1:
            tim2 = 'start'
            maxTag = 'start'
            maxVal = self.bestScoreSubLR(i-1, (tim2, tim1, ti), sentence, cache) + probTiGiven
        else:
            if tim1 in self.tag_set:
                for tag in self.tag_set[tim1]:
                    tim2 = tag
                    maxValNew = max(self.bestScoreSubLR(i-1, (tim2, tim1, ti), sentence, cache) + probTiGiven, maxVal)
                    if (maxValNew > maxVal):
                        maxTag = tag
                    maxVal = maxValNew

        #add to the cache then return the max val
        cache[(i, tim1, ti, tip1)] = (maxVal, maxTag)
        return maxVal


    ########################################################
    # go through set of sentences and return tags
    ########################################################
    def getTagPredictionsL(self, sentence, cache):
        # create the tags list and run the algorithm to get the cache
        # working from the back to the front of the sentence
        tags = []
        ti = 'end'
        tim1 = 'end'
        i = len(sentence) + 1

        #loop through all indices of the cache until we reach the start
        # appends tags to list in reverse order
        while cache.has_key((i, tim1, ti)):
            (maxVal, maxTag) = cache[(i, tim1, ti)]
            tags.append(maxTag)
            tip1 = ti
            ti = tim1
            tim1 = maxTag
            i = i - 1

        # reverse order so the list is from start of sentence to finish
        # and remove the two 'start' tags at the front
        tags.reverse()
        tags.pop(0)
        tags.pop(0)
        #put the tag predictions into the main word feature_vector for analysis
#		import pdb; pdb.set_trace()
        sentence['tag_prediction'] = tags
        if 'tag_prediction' not in self.word_features.columns.values:
            print "Could NOT find tag_prediction column"
            self.word_features['tag_prediction'] = None
        for i in range(len(sentence)):
            self.word_features['tag_prediction'].ix[sentence.ix[i]['index']] = sentence['tag_prediction'].ix[i]

        return tags

    def bestScoreL(self, sentence):
        #for each sentence best score is called, and then it makes this cache
        #the cache stores values for each of the parameters given and can then be used to determine the tokens
        cache = {}

        #let n equal the last index in the list
        n = len(sentence) - 1
        self.bestScoreSubL(n+2, ('end','end'), sentence, cache)

        return cache


    def bestScoreSubL(self, i, (tim1, ti) , sentence, cache):

#        import pdb; pdb.set_trace()
        #takes a cache from best score and adds return values and tags to it
        if cache.has_key((i, tim1, ti)):
            (maxVal, maxTag) = cache[(i, tim1, ti)]
            return maxVal

#        print i, ti

        #left boundry case
        if (i == -1):
            if ((tim1, ti)==('start','start')):
                return 0
            else:
                return NEGINF

        #calculate conditional probability
        if ti == 'end':		#right boundary case
            probTiGiven = 0
        else:
            probTiGiven = self.getConditionalProb(sentence.ix[i], ti) #will be conditional probability of Ti given features P(ti|tim1,tip1,wi)

        #recursive case, loop through all tags, find the tag and val with the maximum likelihood
        maxVal = NEGINF
        maxTag = None
        #Left Boundary: Only check 'start' tag before index 0
        if i <= 1:
            tim2 = 'start'
            maxTag = 'start'
            maxVal = self.bestScoreSubL(i-1, (tim2, tim1), sentence, cache) + probTiGiven
        else:
            if tim1 in self.tag_set:
                for tag in self.tag_set[tim1]:
                    tim2 = tag
                    maxValNew = max(self.bestScoreSubL(i-1, (tim2, tim1), sentence, cache) + probTiGiven, maxVal)
                    if (maxValNew > maxVal):
                        maxTag = tag
                    maxVal = maxValNew

        #add to the cache then return the max val
        cache[(i, tim1, ti)] = (maxVal, maxTag)
        return maxVal
    
    
    ########################################################
    # go through set of sentences and return tags
    ########################################################
    def getTagPredictionsR(self, sentence, cache):
        # create the tags list and run the algorithm to get the cache
        # working from the back to the front of the sentence
        tags = []
        tip1 = 'end'
        ti = 'end'
        i = len(sentence)

#        import pdb; pdb.set_trace()
        #loop through all indices of the cache until we reach the start
        # appends tags to list in reverse order
        while cache.has_key((i, ti, tip1)):
            (maxVal, maxTag) = cache[(i, ti, tip1)]
            tags.append(maxTag)
            tip1 = ti
            ti  = maxTag
            i = i - 1

        # reverse order so the list is from start of sentence to finish
        # and remove the one 'start' tag at the front
        tags.reverse()
        tags.pop(0)
        #put the tag predictions into the main word feature_vector for analysis
#		import pdb; pdb.set_trace()
        sentence['tag_prediction'] = tags
        if 'tag_prediction' not in self.word_features.columns.values:
            print "Could NOT find tag_prediction column"
            self.word_features['tag_prediction'] = None
        for i in range(len(sentence)):
            self.word_features['tag_prediction'].ix[sentence.ix[i]['index']] = sentence['tag_prediction'].ix[i]

        return tags

    def bestScoreR(self, sentence):
        #for each sentence best score is called, and then it makes this cache
        #the cache stores values for each of the parameters given and can then be used to determine the tokens
        cache = {}

        #let n equal the last index in the list
        n = len(sentence) - 1
        self.bestScoreSubR(n+1, ('end','end'), sentence, cache)

        return cache


    def bestScoreSubR(self, i, (ti, tip1) , sentence, cache):

        #takes a cache from best score and adds return values and tags to it
        if cache.has_key((i, ti, tip1)):
            (maxVal, maxTag) = cache[(i, ti, tip1)]
            return maxVal

#        print i, ti

        #left boundry case
        if (i == -1):
            if (ti=='start'):
                return 0
            else:
                return NEGINF

        #calculate conditional probability
        if ti == 'end':		#right boundary case
            probTiGiven = 0
        else:
            probTiGiven = self.getConditionalProb(sentence.ix[i], ti) #will be conditional probability of Ti given features P(ti|tim1,tip1,wi)

        #recursive case, loop through all tags, find the tag and val with the maximum likelihood
        maxVal = NEGINF
        maxTag = None
        #Left Boundary: Only check 'start' tag before index 0
        if i <= 0:
            tim1 = 'start'
            maxTag = 'start'
            maxVal = self.bestScoreSubR(i-1, (tim1, ti), sentence, cache) + probTiGiven
        else:
            if ti in self.tag_set:
                for tag in self.tag_set[ti]:
                    tim1 = tag
                    maxValNew = max(self.bestScoreSubR(i-1, (tim1, ti), sentence, cache) + probTiGiven, maxVal)
                    if (maxValNew > maxVal):
                        maxTag = tag
                    maxVal = maxValNew

        #add to the cache then return the max val
        cache[(i, ti, tip1)] = (maxVal, maxTag)
        return maxVal
