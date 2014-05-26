from POSTagger import POSTagger
import pickle

def train_tagger():
	#Train on brown corpus
	tagger = POSTagger(simplify=True)
	tagger.train(support_cutoff=2)
	nums = tagger.word_features.unique_values()

	return tagger

def test_tagger(tagger, i):
	#Try inference on a sentence
	sentence = tagger.word_features[tagger.word_features['sentence_num'] == i]
	sentence = sentence.reset_index()
	tagger.feature_set = ['t_i-1', 'w_i']
	cache = tagger.bestScoreL(sentence)
	with open("cache" + str(i) + ".p", "wb") as f:
		pickle.dump(cache, f)

	return cache

def load_cache(i):
	with open("cache" + str(i) + ".p", "rb") as f:
		cache = pickle.load(f)
	return cache

#Test the randEval method
def evalTagger(N):
	feature_set = ['t_i-1', 'w_i']
	support_cutoff = 2
	tagger = POSTagger(simplify=True, tag_cutoff=1)
	tagger.randEvalL(N, feature_set, support_cutoff)
    return tagger

