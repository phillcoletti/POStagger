from POSTagger import POSTagger
feature_set = ['w_i']
tagger = POSTagger('None',feature_set,support_cutoff=2,tag_cutoff=1)
tagger.fixedEval()
