# POSTagger
# 5/14/2014
# Phillip Coletti
# Dartmouth College CS-189 Probabilistic Graphical Models (Spring 2014)

import json
import pandas as pd
import numpy as np
import math
import random
import timeit
from nltk.corpus import brown

eps = 10e-50
NEGINF = -10e10

class POSTagger:
	def __init__(self, simplify, tag_cutoff):
	    #Extract features from all sentences
		feature_list = []
		i = 0;
		for sentence in brown.tagged_sents(simplify_tags=simplify):
		    if len(sentence) >= 2:
		        (w1,t1) = sentence.pop(0)
		        (w2,t2) = sentence.pop(0)
		        tags = ['start', 'start', 'start', t1, t2]
		        words = [None, None, None, w1, w2]
		        sentence.append((None, 'end'))
		        sentence.append((None, 'end'))
		        for (w,t) in sentence:
		            #Shift all words and tags down one slot
		            tags.pop(0)
		            tags.append(t)
		            words.pop(0)
		            words.append(w)
		            extra_features = [#(words[1], words[2]), (words[2], words[3]),    #lexical
		                              #(words[2], tags[1]), (words[2], tags[3]),      #lexical-word combo
		                              tags[0] + '_' + tags[1], tags[1] + '_' + tags[3], tags[3] + '_' + tags[4]]  #tag sequence
		            features = [i] + tags + words + extra_features
		            feature_list.append(features)
	            i = i + 1
		#convert to DataFrame
		feature_names = ['sentence_num', 't_i-2', 't_i-1', 't_i', 't_i+1', 't_i+2', 
        				'w_i-2', 'w_i-1', 'w_i', 'w_i+1', 'w_i+2', 
                		#'w_i-1,i', 'w_i,i+1', 'w_i,t_i-1', 'w_i,t_i+1', 
                		't_i-2,i-1', 't_i-1,i+1', 't_i+1,i+2']
		self.word_features = pd.DataFrame(feature_list, columns=feature_names)
		#Calculate Tagset
		frequency = pd.crosstab(rows=self.word_features['t_i'], cols=self.word_features['t_i+1'])
		self.tag_set = {}
		for col in frequency.columns:
		    tags = frequency[col][frequency[col] >= tag_cutoff].index.values
		    self.tag_set[col] = tags

		#Jai's initialization
		# infile_features = open('brown_corpus_prob_features.json', 'r')
		# infile_prior = open('brown_corpus_prob_priors.json', 'r')

		# #load our data structures
		# self.prob_features = json.load(infile_features)
		# self.prob_priors = json.load(infile_prior)
		
		# #lastly load the tag set from the prob_features data structure
		# self.tag_set = self.prob_features["tim1"]

		# infile_features.close()
		# infile_prior.close()

	#####################################################
	# Evaluates the model on a random set of N sentences;
	# Trains on all other sentences
	#####################################################
	def randEvalL(self, N, feature_set, support_cutoff):
		sentence_nums = self.word_features['sentence_num'].unique()
		sample = random.sample(xrange(len(sentence_nums)), N)
		test_set = sentence_nums[sample]
		removeset = set(sample)
		train_set = [v for i, v in enumerate(sentence_nums) if i not in sample]
		print "Training..."
		self.train(train_sentence_nums=train_set, support_cutoff=2)
		print "Testing..."
		self.test(test_sentence_nums=test_set, feature_set=feature_set)


	####################################################################
	# Trains the prior and conditional probabilities
	# prior[t_i] = log_probability(t_i=tag_string)
	# conditional[feature_name][feature_value][t_i] = log_conditional_probability(t_i=tag_string | feature_name=feature_value)
	# If the conditioanl probability is NOT in the conditional dictioanary, it is below the support cutoff; sub in the prior
	#####################################################################
	def train(self, train_sentence_nums, support_cutoff):
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
		    supported = feature_sum[(feature_sum >= support_cutoff)].index
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

	def test(self, test_sentence_nums, feature_set):
		self.feature_set = feature_set
		total_correct = 0;
		total_words = 0;
		for sentence_num in test_sentence_nums:
			sentence = self.word_features[self.word_features['sentence_num'] == sentence_num]
			#set index to begin at 0 for consistency with tag predictions
			sentence = sentence.reset_index()
			tic = timeit.default_timer() 		#start timer
			cache = self.bestScoreL(sentence)
			tags = self.getTagPredictionsL(sentence, cache)
			toc = timeit.default_timer()		#end timer
			correct = sum(sentence['t_i'] == tags)
			words = len(tags)
			avg_time = (toc - tic) / words
			total_correct += correct
			total_words += words
			accuracy = correct / float(words)
			print "Sentence %d: %f" % (sentence_num, accuracy)
			print "%f s avg for %d words" % (avg_time, words)
		total_accuracy = total_correct / float(total_words)
		print "Total Accuracy on %d sentences: %f" % (len(test_sentence_nums), total_accuracy) 

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

	 	#################
		# Jai's old code
		#################
		# conditionalProbVal = 1
		# #conditional for word i
		# if (self.prob_features["w_i"][w_i][ti] != None):
		# 	conditionalProbVal = conditionalProbVal*self.prob_features["w_i"][w_i][ti]
		# else:
		# 	conditionalProbVal = conditionalProbVal*self.prob_priors["w_i"][w_i][ti]

		# #tim1
		# if (self.prob_features["tim1"][tim1][ti] != None):
		# 	conditionalProbVal = conditionalProbVal*self.prob_features["tim1"][tim1][ti]
		# else:
		# 	conditionalProbVal = conditionalProbVal*self.prob_priors["tim1"][tim1][ti]
		
		# #tip1
		# if (self.prob_features["tip1"][tip1][ti] != None):
		# 	conditionalProbVal = conditionalProbVal*self.prob_features["tip1"][tip1][ti]
		# else:
		# 	conditionalProbVal = conditionalProbVal*self.prob_priors["tip1"][tip1][ti]		
		
		return conditionalProbVal


	def returnTokensBi(self, sentence):
		# go through set of sentences and return tags
		
		#create the tokens list and run the algorithm to get the cache
		tokens = []
		sentence_cache = self.bestScore(sentence)

		tokens.append(END)

		#we know what tags we are going to start at and where we will begin extracting from the cache
		tip1 = END
		ti = END
		tim1 = END
		ip1 = len(sentence) + 1

		#loop through all indices of the cache until we reach the start
		while cache.has_key(ip1, (tim1, ti, tip1)):
			(maxVal, maxTag) = cache[ip1, (tim1, ti, tip1)]
			tokens.append(maxTag)
			tip1 = ti
			ti = tim1
			tim1 = maxTag
			ip1 = ip1 - 1

		#returns in reverse order.
		return tokens


	def bestScoreBi(self, sentence):
		#for each sentence best score is called, and then it makes this cache
		#the cache stores values for each of the parameters given and can then be used to determine the tokens
		cache = {}

		#let n equal the last index in the list
		n = len(sentence) - 1
		self.bestScoreSubBi(n+2, (END,END,END), sentence, cache)
		
		return cache


	def bestScoreSubBi(self, ip1, (tim1, ti, tip1) , sentence, cache):
		#takes a cache from best score and adds return values and tags to it

		if cache.has_key(ip1, (tim1, ti, tip1)):
			return cache[ip1, (tim1, ti, tip1)]
		
		i = ip1 - 1

		#left boundry case
		if (i == -1):
			if ((tim1, ti, tip1)==('START','START','START')):
				return 1
			else:
				return 0


		#get the word at index i calculate conditional probability
		w_i = sentence[i]
		probTiGiven = self.getConditionalProb(w_i, tim1, ti, tip1) #will be conditional probability of Ti given features P(ti|tim1,tip1,wi)

		#recursive case, loop through all tags, find the tag and val with the maximum likelihood
		maxVal = 0
		for tag in self.tag_set:
			tim2 = tag
			maxValNew = max(bestScoreSubBi(i, (tim2, tim1, ti), sentence, cache)*probTiGiven, maxVal)
			if (maxValNew != maxVal):
				maxTag = tag
			maxVal = maxValNew

		#add to the cache then return the max val
		cache[ip1,(tim1, ti, tip1)] = (maxVal, maxTag)
		return (maxVal, maxTag)


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
		sentence['tag_prediction'] = tags
		if 'tag_prediction' not in self.word_features.columns.values:
			self.word_features['tag_prediction'] = None
		for i in range(len(sentence)):
			self.word_features.ix[sentence.ix[i]['index']]['tag_prediction'] = sentence.ix[i]['tag_prediction']


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

		#takes a cache from best score and adds return values and tags to it
		if cache.has_key((i, tim1, ti)):
			(maxVal, maxTag) = cache[(i, tim1, ti)]
			return maxVal

		#print i, ti

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



