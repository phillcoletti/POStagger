from POSTagger import POSTagger
feature_set = ['w_i-1', 'w_i', 'w_i+1']
tagger = POSTagger('None', feature_set, 2, 1)
tagger.fixedEval()
