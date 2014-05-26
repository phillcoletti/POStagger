from POSTagger import POSTagger

feature_set = ['t_i-1', 'w_i']
tagger = POSTagger(1)
tagger.fixedEval('L',feature_set, 1)
