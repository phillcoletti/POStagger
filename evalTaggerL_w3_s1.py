from POSTagger import POSTagger
feature_set = ['t_i-1', 'w_i-1', 'w_i', 'w_i+1']
<<<<<<< HEAD
tagger = POSTagger(1)
tagger.fixedEval('L',feature_set, 1)
=======
tagger = POSTagger('L', feature_set, 2, 1)
tagger.fixedEval()
>>>>>>> v2
