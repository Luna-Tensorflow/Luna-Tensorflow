import json

toDel = [
	'Tensorflow.CFunctions',
	'Tensorflow.CWrappers',
	'Tensorflow.Examples',
	'Tensorflow.Main',
	'Tensorflow.Patches',
	'Tensorflow.TestVisual',
	'Tensorflow.Tests',
]

with open('data/doc.json', 'r') as f:
	doc = json.load(f)

	newUnits = []

	for element in doc['units']:
		#print(element, "\n\n\n")
		
		d = False

		for pref in toDel:
			if element['name'].startswith(pref):
				d = True
				break
		
		if not(d):
			newUnits.append(element)	

	
	doc['units'] = newUnits

	with open('data/doc.json', 'w') as f2:
		json.dump(doc, f2)
