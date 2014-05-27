from POSTagger import POSTagger
import sys
feature_set = ['w_i-1', 'w_i', 'w_i+1']
tagger = POSTagger('None', feature_set, support_cutoff=sys.argv[1], 1)
tagger.fixedEval()
