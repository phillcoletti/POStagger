

def getConitionalProb(i, tim1, ti, tip1, feature_set):
	


def bestScore(n, tag_set, feature_set):
	cache = {}
	return bestScoreSub(n+2, (END,END,END), tag_set, cache)


def bestScoreSub(ip1, (tim1, ti, tip1), tag_set, feature_set, cache):
	

	if cache.has_key(ip1, (tim1, ti, tip1)):
		return cache[ip1, (tim1, ti, tip1)]
	
	i = ip1 - 1

	#left boundry case
	if (i == -1):
		if ((tim1, ti, tip1)==(START,START,START)):
			return 1
		else:
			return 0



	probTiGiven = getConditionalProb(i, tim1, ti, tip1, feature_set) #will be conditional probability of Ti given features P(ti|tim1,tip1,wi)


	#recursive case
	maxVal = 0
	for tag in tag_set:
		tim2 = tag
		maxVal = max(bestScoreSub(i, (tim2, tim1, ti), tag_set, cache)*probTiGiven, maxVal)


	#add to the cache then return the max val
	cache[ip1,(tim1, ti, tip1)] = maxVal
	return maxVal