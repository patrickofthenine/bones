from owlready2 import *
import os
namespace.set_log_level(1)

class Owl:
	def load_config(self):
		return 'load_config'

	def get_ontology_location(self):
		dir = os.getcwd()
		onto_path.append(dir + '/training_images/owls/fma.owl')
		return onto_path

	def get_onto(self):
		onto_loc 	= self.get_ontology_location()
		onto 		= get_ontology(onto_loc[0]).load()
		return onto

	def run_once(self):
		ontology = self.get_onto()
		ontology_classes = list(ontology.classes())
		for onto_class in ontology_classes:

O = Owl()
onto = O.run_once()


