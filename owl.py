from owlready2 import *
import os

class Owl:
	def load_config(self):
		return 'load_config'

	def get_ontology_location(self):
		dir = os.getcwd()
		onto_path.append(dir + '\\training_images\\owls\\fma.owl')
		return onto_path

	def get_onto(self):
		onto_loc 	= self.get_ontology_location()
		onto 		= get_ontology(onto_loc[0])
		onto  		= onto.load()
		return onto

	def run_once(self):
		print('namin space', namespace)
		namespace.set_log_level(1)
		ontology = self.get_onto()
		#with ontology:
		#	print('onto', ontology)
		#onto_loc = self.get_ontology_location()
		#print(onto_loc)
		#onto_path = [].clear()

O = Owl()
onto = O.run_once()


