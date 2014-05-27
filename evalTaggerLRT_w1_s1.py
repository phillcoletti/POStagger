from POSTagger import POSTagger
feature_set = ['t_i-1,i+1','t_i-1', 't_i+1', 'w_i']
tagger = POSTagger('LR',feature_set,support_cutoff=2,tag_cutoff=1)
tagger.fixedEval()
