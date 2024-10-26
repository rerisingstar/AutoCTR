import ConfigSpace
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter


def get_hpo_cs(space, algo):
	BaseHPs, SearchSpace, ModelUse = space
	algo_cs = ConfigurationSpace()
	for hp in ModelUse[algo]:
		if isinstance(SearchSpace[hp][0], list):
			hp_dict = ["list@"+str(i) for i in range(len(SearchSpace[hp]))]
		else:
			hp_dict = SearchSpace[hp]
		algo_cs.add_hyperparameter(CategoricalHyperparameter(algo+","+hp, hp_dict))
	for hp in BaseHPs.keys():
		if isinstance(SearchSpace[hp][0], list):
			hp_dict = ["list@"+str(i) for i in range(len(SearchSpace[hp]))]
		else:
			hp_dict = SearchSpace[hp]
		algo_cs.add_hyperparameter(CategoricalHyperparameter(algo+","+hp, hp_dict))
	return algo_cs

def get_configuration_space(space):
	BaseHPs, SearchSpace, ModelUse = space

	cs = ConfigurationSpace()
	estimators = list(ModelUse.keys())
	algo = CategoricalHyperparameter('algorithm', estimators)
	cs.add_hyperparameter(algo)

	for estimator in estimators:
		estimator_cs = get_hpo_cs(space, estimator)
		parent_hyperparameter = {'parent': algo,
								'value': estimator}
		cs.add_configuration_space(estimator, estimator_cs, parent_hyperparameter=parent_hyperparameter)
	return cs

def get_configuration_space_module(space, estimator):
	BaseHPs, SearchSpace, ModelUse = space

	cs = ConfigurationSpace()
	algo = CategoricalHyperparameter('estimator', [estimator], default_value=estimator)
	cs.add_hyperparameter(algo)

	estimator_cs = get_hpo_cs(space, estimator)
	parent_hyperparameter = {'parent': algo,
							'value': estimator}
	cs.add_configuration_space(estimator, estimator_cs, parent_hyperparameter=parent_hyperparameter)
	return cs
