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

def evalTaggerL_w1_s1():
    feature_set = ['t_i-1', 'w_i']
    tagger = POSTagger(1)
    tagger.fixedEval('L',feature_set, 1)

def evalTaggerL_w3_s1():
    feature_set = ['t_i-1', 'w_i-1', 'w_i', 'w_i+1']
    tagger = POSTagger(1)
    tagger.fixedEval('L',feature_set, 1)

def evalTaggerR_w3_s1():
    feature_set = ['t_i+1', 'w_i-1', 'w_i', 'w_i+1']
    tagger = POSTagger(1)
    tagger.fixedEval('R',feature_set, 1)

def evalTaggerR_w1_s1():
    feature_set = ['t_i+1', 'w_i']
    tagger = POSTagger(1)
    tagger.fixedEval('R',feature_set, 1)

def evalTaggerLR_w3_s1():
    feature_set = ['t_i-1', 't_i+1', 'w_i-1', 'w_i', 'w_i+1']
    tagger = POSTagger(1)
    tagger.fixedEval('LR',feature_set, 1)

def evalTaggerLR_w1_s1():
    feature_set = ['t_i-1', 't_i+1', 'w_i']
    tagger = POSTagger(1)
    tagger.fixedEval('LR',feature_set, 1)

def evalTaggerLRT_w3_s1():
    feature_set = ['t_i-1,i+1', 't_i-1', 't_i+1', 'w_i-1', 'w_i', 'w_i+1']
    tagger = POSTagger(1)
    tagger.fixedEval('LR',feature_set, 1)

def evalTaggerLRT_w1_s1():
    feature_set = ['t_i-1,i+1','t_i-1', 't_i+1', 'w_i']
    tagger = POSTagger(1)
    tagger.fixedEval('LR',feature_set, 1)
