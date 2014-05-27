from POSTagger import POSTagger
<<<<<<< HEAD

feature_set = ['t_i-1', 'w_i']
tagger = POSTagger(1)
tagger.fixedEval('L',feature_set, 1)
=======
feature_set = ['t_i-1', 'w_i']
tagger = POSTagger('L',feature_set,support_cutoff=2,tag_cutoff=1)
tagger.fixedEval()
>>>>>>> v2
