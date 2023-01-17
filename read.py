import pandas as pd
from utils import *
from myModel import *
from datasets import Dataset
import joblib

descs = ['AllDesc',
		 # 'MoleculeDesc',
		 # 'MoleculeAndPeptideDesc'
		 ]

tasks = [  # 'mouse_blood_modification',
	# 'mouse_blood_nature',
	# 'mouse_crude_intestinal_modification',
	'Human_blood_nature',
	# 'Human_blood_modification',
]

noDescriptorsField = ['SMILES', 'label', 'SEQUENCE']

model=joblib.load('./model/AllDesc_Human_blood_nature_SVR_log2_kfold.pkl')
print(model)