from modlamp.descriptors import *
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
from rdkit import Chem
from PyBioMed.PyMolecule import basak, bcut, cats2d, charge, connectivity, constitution, \
	estate, geary
from PyBioMed.PyMolecule import kappa, moe, molproperty, moran, moreaubroto, topology
from PyBioMed import Pyprotein


def getGlobalDescriptor(seqList, amide):
	des = GlobalDescriptor(seqList)
	# 这里C末端酰胺化需要设置
	des.calculate_all(amide=amide)
	res = {}
	desT = des.descriptor.T
	for i, name in enumerate(des.featurenames):
		res["modlAMP" + name] = desT[i]  # 'MW'会有歧义
	return res


# fingerprint 不能成功导入
def _getPymoleculeDescriptorFromSmi(smi):
	res = {}
	mol = Chem.MolFromSmiles(smi)
	res = {}
	res.update(kappa.GetKappa(mol))
	res.update(charge.GetCharge(mol))
	res.update(connectivity.GetConnectivity(mol))
	# res.update(constitution.GetConstitutional(mol))  constitution需要依赖openbabel
	res.update(estate._GetEstate(mol))
	#res.update(geary.GetGearyAuto(mol)) 'key error *'
	res.update(moe.GetMOE(mol))
	#res.update(moran.GetMoranAuto(mol)) 'key error *'
	res.update(moreaubroto.GetMoreauBrotoAuto(mol))
	# res.update(topology.GetTopology(mol)) #runtime warning
	res.update(molproperty.GetMolecularProperty(mol))
	res.update(basak.Getbasak(mol))
	res.update(bcut.GetBurden(mol))
	res.update(cats2d.CATS2D(mol, PathLength=10, scale=3))
	
	return res


def getPymoleculeDescriptorFromSmi(smiList):
	dicList = []
	for smi in smiList:
		dicList.append(_getPymoleculeDescriptorFromSmi(smi))
	keys = dicList[0].keys()
	res = defaultdict(list)
	for key in keys:
		for dic in dicList:
			res['Pymolecule' + key].append(dic[key])
	return res


def _getPyproteinDescriptorFromSeq(seq):
	protein_class = Pyprotein.PyProtein(seq)
	res = {}
	res.update(protein_class.GetAAComp())  # 20
	#res.update(protein_class.GetDPComp())  # 400		#bug
	# res.update(protein_class.GetTPComp())               #8000
	#res.update(protein_class.GetMoreauBrotoAuto())  # 240 #除0bug
	# res.update(protein_class.GetMoranAuto())  # 240	`#除0bug
	# res.update(protein_class.GetGearyAuto())  # 240	 #除0bug
	# res.update(protein_class.GetCTD())  # 147			  #除0bug
	# res.update(protein_class.GetPAAC())                 #除0bug
	# res.update(protein_class.GetAPAAC())                #除0bug
	#res.update(protein_class.GetSOCN())  # 90
	#res.update(protein_class.GetQSO())  # 100				#除0bug
	res.update(protein_class.GetTriad())  # 343
	
	return res


def getPyproteinDescriptorFromSeq(seqList):
	dicList = []
	for seq in seqList:
		dicList.append(_getPyproteinDescriptorFromSeq(seq))
	keys = dicList[0].keys()
	res = defaultdict(list)
	for key in keys:
		for dic in dicList:
			res['Pyprotein' + key].append(dic[key])
	return res


def isAmidation(smiList):
	res = []
	for smi in smiList:
		mol = Chem.MolFromSmiles(smi)
		amidationPatt = Chem.MolFromSmarts('C(=O)[NH][CH]C(=O)[NH2]')
		match = mol.GetSubstructMatch(amidationPatt)
		if (len(match) != 0):
			res.append(True)
		else:
			res.append(False)
	return res


def isAcetylation(smiList):
	res = []
	for smi in smiList:
		mol = Chem.MolFromSmiles(smi)
		acetylationPatt = Chem.MolFromSmarts('[CH3]C(=O)[NH][CH]C(=O)')
		match = mol.GetSubstructMatch(acetylationPatt)
		if (len(match) != 0):
			res.append(True)
		else:
			res.append(False)
	return res



def getEnzyData(seqList):
	res = defaultdict(list)
	
	EnzyList = ['Arg-C proteinase', 'Asp-N endopeptidase', 'Asp-N endopeptidase + N-terminal Glu', 'BNPS-Skatole',
				'Caspase1', 'Caspase2',
				'Caspase3', 'Caspase4', 'Caspase5', 'Caspase6', 'Caspase7', 'Caspase8', 'Caspase9', 'Caspase10',
				'Chymotrypsin-high specificity (C-term to [FYW], not before P)',
				'Chymotrypsin-low specificity (C-term to [FYWML], not before P)',
				'Clostripain', 'CNBr', 'Enterokinase', 'GranzymeB', 'Factor Xa', 'Formic acid',
				'Glutamyl endopeptidase', 'Hydroxylamine', 'Iodosobenzoic acid',
				'LysC', 'LysN', 'NTCB (2-nitro-5-thiocyanobenzoic acid)', 'Pepsin (pH1.3)', 'Pepsin (pH>2)',
				'Proline-endopeptidase', 'Proteinase K',
				'Staphylococcal peptidase I', 'Tobacco etch virus protease', 'Thermolysin', 'Thrombin', 'Trypsin']
	
	headers = {'User-Agent': 'Mozilla/5.0'}
	
	for seq in seqList:
		seqRes = {}
		payload = {'protein': seq, 'enzyme_number': 'all_enzymes', 'cleave_number': 'all', 'alphtable': 'alphtable'}
		session = requests.Session()
		response = session.post("https://web.expasy.org/cgi-bin/peptide_cutter/peptidecutter.pl/", headers=headers,
								data=payload)
		
		soup = BeautifulSoup(response.content, "lxml")
		try:
			table = soup.findAll("table", class_="proteomics2")[0]
			for real_rows in table.findAll("tr"):
				td = [i.text.replace("\n", "") for i in real_rows.findAll("td")]
				if td != []: seqRes[td[0]] = td[1:]
			
			# 下面对每一个多肽计算酶描述符
			# 将字符串转换成数字,单个酶的所有位点信息保存在allEnzyInfoList
			
			allEnzyInfoList = []
			for key in seqRes.keys():
				value = []
				sites = seqRes[key][1].split()
				for site in sites:
					value.append(int(site))
					allEnzyInfoList.append(int(site))
				seqRes[key] = value
			
			# print(allEnzyInfoList)  #有多个酶有同一切割位点,下面是 AFDGHLKI 的 allEnzyInfoList
			# [2, 2, 2, 2, 5, 6, 3, 7, 6, 1, 2, 5, 6, 1, 2, 5, 6, 1, 2, 6, 8, 1, 5, 7, 7]
			# print(seqRes)
			# print(seq)
			
			# 酶切总数暂时定义为37个酶中能对给定多肽产生切割的酶的数目
			numSites = len(seqRes)
			res['numSites'].append(numSites)
			# 最近酶切位点暂时定义为切割位点的最近距离,即allEnzyInfoList中两两元素的最近距离
			nearestSites = 1e6
			for i in allEnzyInfoList:
				for j in allEnzyInfoList:
					if (abs(i - j) < nearestSites): nearestSites = abs(i - j)
					if (abs(i - j) == 0): break  # 0是最小的情况,只要找到0就可以停止
			res['nearestSites'].append(nearestSites)
			# 酶切位点权重暂时定义为能被给定酶切割位点的数目
			for enzy in EnzyList:
				if enzy in seqRes:
					res[enzy].append(len(seqRes[enzy]))
				else:
					res[enzy].append(0)
		except:
			print("*" * 20)
			print("{} 出现问题".format(seq))
			print("*" * 20)
			res['numSites'].append(0)
			res['nearestSites'].append(0)
			for enzy in EnzyList: res[enzy].append(0)
	
	return res


def getMoleculeDesc(seqList, smiList, acetylation=None, amidation=None):
	if (amidation == None): amidation = isAmidation(smiList)
	if (acetylation == None): acetylation = isAcetylation(smiList)
	
	res = {'amidation': amidation, 'acetylation': acetylation}
	
	res.update(getPymoleculeDescriptorFromSmi(smiList))
	res.update(getGlobalDescriptor(seqList, amidation))
	
	return res


def getMoleculeAndPeptideDesc(seqList, smiList, acetylation=None, amidation=None):
	res = getPyproteinDescriptorFromSeq(seqList)
	res.update(getMoleculeDesc(seqList, smiList, acetylation, amidation))
	return res


def getAllDesc(seqList, smiList, acetylation=None, amidation=None):
	res = getEnzyData(seqList)
	res.update(getMoleculeAndPeptideDesc(seqList, smiList, acetylation, amidation))
	return res
