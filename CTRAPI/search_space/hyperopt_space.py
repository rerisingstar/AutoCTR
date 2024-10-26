import hyperopt
from hyperopt import hp


def get_hpo_cs(space, algo):
	BaseHPs, SearchSpace, ModelUse = space
	hp_space = dict()
	for hyper_parameter in ModelUse[algo]:
		hp_space[algo+","+hyper_parameter] = hp.choice(algo+","+hyper_parameter, SearchSpace[hyper_parameter])
	for hyper_parameter in BaseHPs.keys():
		hp_space[algo+","+hyper_parameter] = hp.choice(algo+","+hyper_parameter, SearchSpace[hyper_parameter])
	return hp_space

def get_hyperopt_space(space):
	BaseHPs, SearchSpace, ModelUse = space

	cs = dict()
	estimators = list(ModelUse.keys())
	for estimator in estimators:
		estimator_cs = get_hpo_cs(space, estimator)
		cs[estimator] = estimator_cs

	config_space = {'estimator': hp.choice('estimator', [(estimator, cs[estimator]) for estimator in estimators])}
	return estimators, config_space
