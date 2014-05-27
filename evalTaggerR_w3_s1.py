from POSTagger import POSTagger
feature_set = ['t_i+1', 'w_i-1', 'w_i', 'w_i+1']
tagger = POSTagger('R', feature_set, 2, 1)
tagger.fixedEval()
