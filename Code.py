#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:49:56 2018

@author: gabrielefrattaroli
"""
import pandas as pd
import numpy as np

Private_Schools_DB = pd.read_csv('SCUANAGRAFEPAR20171820170901.csv')
Public_Schools_DB = pd.read_csv('SCUANAGRAFESTAT20171820170901.csv')


##### Dropping different columns between eachother

Public_Schools_DB = Public_Schools_DB.drop(['CODICEISTITUTORIFERIMENTO'], axis=1)
Public_Schools_DB = Public_Schools_DB.drop(['DENOMINAZIONEISTITUTORIFERIMENTO'], axis=1)
Public_Schools_DB = Public_Schools_DB.drop(['DESCRIZIONECARATTERISTICASCUOLA'], axis=1)
Public_Schools_DB = Public_Schools_DB.drop(['INDICAZIONESEDEDIRETTIVO'], axis=1)
Public_Schools_DB = Public_Schools_DB.drop(['INDICAZIONESEDEOMNICOMPRENSIVO'], axis=1)
Public_Schools_DB = Public_Schools_DB.drop(['SEDESCOLASTICA'], axis=1)

##### Binding the two DB into one

Schools_DB = Public_Schools_DB.append(pd.DataFrame(data = Private_Schools_DB), ignore_index=True)

##### Removing columns that are not referencing location

Schools_DB = Schools_DB.drop(['ANNOSCOLASTICO'], axis=1)
Schools_DB = Schools_DB.drop(['CODICESCUOLA'], axis=1)
Schools_DB = Schools_DB.drop(['DESCRIZIONETIPOLOGIAGRADOISTRUZIONESCUOLA'], axis=1)
Schools_DB = Schools_DB.drop(['INDIRIZZOEMAILSCUOLA'], axis=1)
Schools_DB = Schools_DB.drop(['INDIRIZZOPECSCUOLA'], axis=1)
Schools_DB = Schools_DB.drop(['SITOWEBSCUOLA'], axis=1)

##### Area Geografica is a step-up from region so we don't need it, also the name of the school

Schools_DB = Schools_DB.drop(['AREAGEOGRAFICA'], axis=1)
Schools_DB = Schools_DB.drop(['DENOMINAZIONESCUOLA'], axis=1)


###### DB is ready

###### Assigining numbers to categories

Regions = np.unique(Schools_DB['REGIONE']).tolist()
RegionsCat = range(len(Regions))
RegionsCat_dict = {}
for i in range(len(Regions)):
    RegionsCat_dict[Regions[i]] = RegionsCat[i]
Schools_DB['RegionsCat'] = Schools_DB['REGIONE'].copy()
Schools_DB = Schools_DB.replace({'RegionsCat': RegionsCat_dict})


Provincia = np.unique(Schools_DB['PROVINCIA']).tolist()
ProvinciaCat = range(len(Provincia))
ProvinciaCat_dict = {}
for i in range(len(Provincia)):
    ProvinciaCat_dict[Provincia[i]] = ProvinciaCat[i]
Schools_DB['ProvinciaCat'] = Schools_DB['PROVINCIA'].copy()
Schools_DB = Schools_DB.replace({'ProvinciaCat': ProvinciaCat_dict})


Town = np.unique(Schools_DB['DESCRIZIONECOMUNE']).tolist()
TownCat = range(len(Town))
TownCat_dict = {}
for i in range(len(Town)):
    TownCat_dict[Town[i]] = TownCat[i]
Schools_DB['TownCat'] = Schools_DB['DESCRIZIONECOMUNE'].copy()
Schools_DB = Schools_DB.replace({'TownCat': TownCat_dict})


TownCode = np.unique(Schools_DB['CODICECOMUNESCUOLA']).tolist()
TownCodeCat = range(len(TownCode))
TownCodeCat_dict = {}
for i in range(len(TownCode)):
    TownCodeCat_dict[TownCode[i]] = TownCodeCat[i]
Schools_DB['TownCodeCat'] = Schools_DB['CODICECOMUNESCUOLA'].copy()
Schools_DB = Schools_DB.replace({'TownCodeCat': TownCodeCat_dict})


Schools_DBtrain = Schools_DB.drop(['RegionsCat'], axis=1)
Schools_DBtrain = Schools_DBtrain.drop(['PROVINCIA'], axis=1)
Schools_DBtrain = Schools_DBtrain.drop(['INDIRIZZOSCUOLA'], axis=1)
Schools_DBtrain = Schools_DBtrain.drop(['CODICECOMUNESCUOLA'], axis=1)
Schools_DBtrain = Schools_DBtrain.drop(['DESCRIZIONECOMUNE'], axis=1)
Schools_DBtrain = Schools_DBtrain.drop(['REGIONE'], axis=1)
mapping = {'Non Disponibile': 0,'-----':00,'.':000}
Schools_DBtrain = Schools_DBtrain.replace({'CAPSCUOLA': mapping})


###### Decision Tree
import time

t0 = time.time()
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Schools_DBtrain, Schools_DB['RegionsCat'])
from sklearn.model_selection import cross_val_score
results = cross_val_score(clf, Schools_DBtrain, Schools_DB['RegionsCat'], cv=10)
print (results)
t1 = time.time()
total = t1-t0
print (total)

# feature selection - Univariate
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=2).fit_transform(Schools_DBtrain, Schools_DB['RegionsCat'])
X_new.shape




t0 = time.time()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_new, Schools_DB['RegionsCat'])
results = cross_val_score(clf, X_new, Schools_DB['RegionsCat'], cv=10)
print (results)
t1 = time.time()
total = t1-t0
print (total)

###### SVM
#t0 = time.time()
#from sklearn import svm
#clf = svm.SVC()
#clf.fit(Schools_DBtrain, Schools_DB['RegionsCat'])  
#results = cross_val_score(clf, Schools_DBtrain, Schools_DB['RegionsCat'], cv=10)
#print (results) 
#t1 = time.time()
#total = t1-t0

#t0 = time.time()
#from sklearn import svm
#clf = svm.SVC()
#clf.fit(X_new, Schools_DB['RegionsCat'])  
#results = cross_val_score(clf, X_new, Schools_DB['RegionsCat'], cv=10)
#print (results) 
#t1 = time.time()
#total = t1-t0

##### Random Forest
t0 = time.time()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=3, random_state=0)  ##### Original DB
clf.fit(Schools_DBtrain, Schools_DB['RegionsCat'])  
results = cross_val_score(clf, Schools_DBtrain, Schools_DB['RegionsCat'], cv=10)
print (results) 
t1 = time.time()
total = t1-t0
print (total)

t0 = time.time()
clf = RandomForestClassifier(max_depth=3, random_state=0)  ##### Feature-selected DB
clf.fit(X_new, Schools_DB['RegionsCat'])  
results = cross_val_score(clf, X_new, Schools_DB['RegionsCat'], cv=10)
print (results) 
t1 = time.time()
total = t1-t0
print (total)

##### Random Forest Depth 10
t0 = time.time()
clf = RandomForestClassifier(max_depth=10, random_state=0)  ##### Original DB
clf.fit(Schools_DBtrain, Schools_DB['RegionsCat'])  
results = cross_val_score(clf, Schools_DBtrain, Schools_DB['RegionsCat'], cv=10)
print (results) 
t1 = time.time()
total = t1-t0
print (total)

t0 = time.time()
clf = RandomForestClassifier(max_depth=10, random_state=0)  ##### Feature-selected DB
clf.fit(X_new, Schools_DB['RegionsCat'])  
results = cross_val_score(clf, X_new, Schools_DB['RegionsCat'], cv=10)
print (results) 
t1 = time.time()
total = t1-t0
print (total)

##### Regression Tree
t0 = time.time()
clf = tree.DecisionTreeRegressor()
clf.fit(Schools_DBtrain, Schools_DB['RegionsCat'])  
results = cross_val_score(clf, Schools_DBtrain, Schools_DB['RegionsCat'], cv=10)
print (results)
t1 = time.time()
total = t1-t0
print (total)

t0 = time.time()
clf = tree.DecisionTreeRegressor()
clf.fit(X_new, Schools_DB['RegionsCat'])  
results = cross_val_score(clf, X_new, Schools_DB['RegionsCat'], cv=10)
print (results)
t1 = time.time()
total = t1-t0
print (total)
###### Histograms

import matplotlib.pyplot as plt

plt.title("Regions Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.hist(Schools_DB['RegionsCat'])
plt.show()

plt.title("Provincia Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.hist(Schools_DB['ProvinciaCat'])
plt.show()

plt.title("Town Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.hist(Schools_DB['TownCat'])
plt.show()

plt.title("TownCode Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.hist(Schools_DB['TownCodeCat'])
plt.show()

plt.title("CAPSCUOLA Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.hist(Schools_DB['CAPSCUOLA'], bins=20)
plt.show()


