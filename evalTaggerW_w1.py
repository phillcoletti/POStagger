from POSTagger import POSTagger
import sys

feature_set = ['w_i']
tagger = POSTagger('None',feature_set,support_cutoff=sys.argv[1],tag_cutoff=1)
tagger.fixedEval()
