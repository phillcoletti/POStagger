


class POSTagger:
	def __init__(self):
		import json
		
		infile_features = open('brown_corpus_prob_features.json', 'r')
		infile_prior = open('brown_corpus_prob_priors.json', 'r')

		#load our data structures
		self.prob_features = json.load(infile_features)
		self.prob_priors = json.load(infile_prior)
		
		#lastly load the tag set from the prob_features data structure
		self.tag_set = self.prob_features["tim1"]

		infile_features.close()
		infile_prior.close()

	def getConitionalProb(self, w_i, tim1, ti, tip1):
		#if the conditional probability is equal to None, then we assign a prior
		conditionalProbVal = 1

		#conditional for word i
		if (self.prob_features["w_i"][w_i][ti] != None):
			conditionalProbVal = conditionalProbVal*self.prob_features["w_i"][w_i][ti]
		else:
			conditionalProbVal = conditionalProbVal*self.prob_priors["w_i"][w_i][ti]

		#tim1
		if (self.prob_features["tim1"][tim1][ti] != None):
			conditionalProbVal = conditionalProbVal*self.prob_features["tim1"][tim1][ti]
		else:
			conditionalProbVal = conditionalProbVal*self.prob_priors["tim1"][tim1][ti]
		
		#tip1
		if (self.prob_features["tip1"][tip1][ti] != None):
			conditionalProbVal = conditionalProbVal*self.prob_features["tip1"][tip1][ti]
		else:
			conditionalProbVal = conditionalProbVal*self.prob_priors["tip1"][tip1][ti]		
		
		return conditionalProbVal


	def returnTokens(self, sentence):
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


	def bestScore(self, sentence):
		#for each sentence best score is called, and then it makes this cache
		#the cache stores values for each of the parameters given and can then be used to determine the tokens
		cache = {}

		#let n equal the last index in the list
		n = len(sentence) - 1


		self.bestScoreSub(n+2, (END,END,END), sentence, cache)
		
		return cache


	def bestScoreSub(self, ip1, (tim1, ti, tip1) , sentence, cache):
		#takes a cache from best score and adds return values and tags to it

		if cache.has_key(ip1, (tim1, ti, tip1)):
			return cache[ip1, (tim1, ti, tip1)]
		
		i = ip1 - 1

		#left boundry case
		if (i == -1):
			if ((tim1, ti, tip1)==(START,START,START)):
				return 1
			else:
				return 0


		#get the word at index i calculate conditional probability
		w_i = sentence[i]
		probTiGiven = self.getConditionalProb(w_i, tim1, ti. tip1) #will be conditional probability of Ti given features P(ti|tim1,tip1,wi)

		#recursive case, loop through all tags, find the tag and val with the maximum likelihood
		maxVal = 0
		for tag in self.tag_set:
			tim2 = tag
			maxValNew = max(bestScoreSub(i, (tim2, tim1, ti), cache)*probTiGiven, maxVal)
			if (maxValNew != maxVal):
				maxTag = tag
			maxVal = maxValNew

		#add to the cache then return the max val
		cache[ip1,(tim1, ti, tip1)] = (maxVal, maxTag)
		return (maxVal, maxTag)



