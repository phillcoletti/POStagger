from POSTagger import POSTagger
feature_set = ['t_i-1', 't_i+1', 'w_i']
tagger = POSTagger('LR', feature_set, 2, 1)
tagger.fixedEval()
