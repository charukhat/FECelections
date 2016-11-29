__author__ = 'charukhatwani'


#Array processing
import subprocess
import numpy as np

from sklearn import datasets
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.cross_validation import StratifiedKFold
#Data analysis, wrangling and common exploratory operations

import pandas as pd
from pandas import Series, DataFrame

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns

# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

import pandas as pd


fec = pd.read_csv('P00000001-OH-2008.csv',index_col=False)
fec_2012=pd.read_csv('P00000001-OH-2012.csv',index_col=False)
fec_2016=pd.read_csv('P00000001-ALL-2016.csv',index_col=False)
print fec.info()

unique_cands_2008= fec.cand_nm.unique()
unique_cands_2012= fec_2012.cand_nm.unique()
unique_cands_2016= fec_2016.cand_nm.unique()

print '2008 candidates'
print unique_cands_2008
print '2012 candidates'
print unique_cands_2012
print '2016 candidates'
print unique_cands_2016


#while doing analysis later we want to associate each candidate with their party
parties = {'Bachmann, Michele': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democratic',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican',
           'Stein, Jill':'Green',
            'Rubio, Marco':'Republican',
            'Santorum, Richard J.':'Republican',
            'Perry, James R. (Rick)':'Republican',
            'Carson, Benjamin S.':'Republican',
            "Cruz, Rafael Edward 'Ted'":'Republican',
            'Paul, Rand':'Republican',
            'Clinton, Hillary Rodham':'Democratic',
            'Sanders, Bernard':'Democratic',
            'Fiorina, Carly':'Republican',
            'Huckabee, Mike':'Republican',
            'Pataki, George E.':'Republican',
            "O'Malley, Martin Joseph":'Democratic',
            'Graham, Lindsey O.':'Republican',
            'Bush, Jeb':'Republican',
            'Trump, Donald J.':'Republican',
            'Jindal, Bobby':'Republican',
            'Christie, Christopher J.':'Republican',
            'Walker, Scott':'Republican',
            'Webb, James Henry Jr.':'Republican',
            'Kasich, John R.':'Republican',
            'Lessig, Lawrence':'Democratic',
            'Huckabee, Mike':	'Republican',
            'Paul, Ron':	'Republican',
            'Hunter, Duncan':	'Republican',
            'Thompson, Fred Dalton':	'Republican',
            'Richardson, Bill':	'Democratic',
            'McCain, John S':	'Republican',
            'Clinton, Hillary Rodham':	'Democratic',
            'Edwards, John':	'Democratic',
            'Giuliani, Rudolph W':	'Republican',
            'Brownback, Samuel Dale':'Republican',
            'Tancredo, Thomas Gerald':	'Republican',
            'Cox, John H':'Republican',
            'Biden, Joseph R Jr':	'Democratic',
            'Gravel, Mike':	'Democratic',
            'Dodd, Christopher J':'Democratic',
            'Kucinich, Dennis J':'Democratic',
            'Gilmore, James S III':'Republican',
            '':'Other'}

# Adding a column party that sets value to the party candidates
# The way this line works is as follows:
#  1. fec_all.cand_nm gives a vector (or Series in Pandas terminology)
#  2. For each row, the code looks up the candidate name to the dictionary parties
#  3. If the name of the candidate (cand_nm) is in parties, it returns the value (i.e. Republican or Democrat)
#  4. This whole thing is done for each row and you get another vector as output
#  5. Finally, we create a new column in fec_all called 'party' and assign the vector
#print fec.cand_nm[123456:123461]

fec['party'] = fec.cand_nm.map(parties)
fec_2012['party'] = fec_2012.cand_nm.map(parties)
fec_2016['party'] = fec_2016.cand_nm.map(parties)

#Calculating the counts of each year party wise

print '2008 party wise vote counts'
print fec['party'].value_counts()

print '2012 party wise vote counts'
print fec_2012['party'].value_counts()

print '2016 party wise vote counts'
print fec_2016['party'].value_counts()

#To simplify analysis I will restrict only to positive contributions
fec = fec[fec.contb_receipt_amt > 0]
fec_2012 = fec_2012[fec_2012.contb_receipt_amt > 0]
fec_2016 = fec_2016[fec_2016.contb_receipt_amt > 0]

fec_2012['party'] = fec_2012.cand_nm.map(parties)
fec_2016['party'] = fec_2016.cand_nm.map(parties)


#Top 10 Contribution Statistics by Occupation


Contribution_2008=fec.contbr_occupation.value_counts()[:10]
print '2008 top 10 contributions by occupation'
print Contribution_2008


Contribution_2012=fec_2012.contbr_occupation.value_counts()[:10]
print '2012 top 10 contributions by occupation'
print Contribution_2012


Contribution_2016=fec_2016.contbr_occupation.value_counts()[:10]
print '2016 top 10 contributions by occupation'
print Contribution_2016

#Now we see the bifurcation donation party wise
#import pandas.util.testing as tm; tm.N = 3

by_occupation = fec.pivot_table('contb_receipt_amt', index='contbr_occupation',
columns='party',aggfunc=np.sum)

over_1mm_2008 = by_occupation[by_occupation.sum(1) > 1000000]

print '2008 over 1 million contributions'
print over_1mm_2008

by_occupation = fec_2012.pivot_table('contb_receipt_amt', index='contbr_occupation',
columns='party',aggfunc=np.sum)

over_1mm_2012 = by_occupation[by_occupation.sum(1) > 1000000]

print '2012 over 1 million contributions'
print over_1mm_2012

by_occupation = fec_2016.pivot_table('contb_receipt_amt', index='contbr_occupation',
columns='party',aggfunc=np.sum)



over_1mm_2016 = by_occupation[by_occupation.sum(1) > 1000000]

print '2016 over 1 million contributions'
print over_1mm_2016


def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
# Order totals by key in descending order return totals.order(ascending=False)[-n:]

#Calculating the top 10 occupations
grouped = fec.groupby('cand_nm')
print grouped.apply(get_top_amounts, 'contbr_occupation', n=5)

#Donations by state

grouped = fec.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
print totals[:10]

by_occupation = fec.pivot_table('contb_receipt_amt', index='contbr_zip',
columns='party',aggfunc=np.sum)

over_1mm_2008 = by_occupation[by_occupation.sum(1) > 1000000]

print over_1mm_2008

#We tend to use zip code as a categorial variable and trimming the zip value to contain only 5 characters

fec["zip_valid"]=(fec.contbr_zip).map(lambda x:str(x)[:5])
fec["zip_valid"]=(fec.contbr_zip).map({'NG7 1': 12345,'V6R 3M9': 12345,'M5T 2': 12345,'4221s':12345})
print fec.zip_valid[fec.zip_valid.isin(['M5T 2'])]

fec["zip_valid"]=pd.DataFrame(fec["zip_valid"].astype(float))
fec["zip_valid"]=fec["zip_valid"].fillna(value=0)


fec_2012["zip_valid"]=(fec_2012.contbr_zip).map(lambda x:str(x)[:5])
fec_2012["zip_valid"]=(fec_2012.contbr_zip).map({'NG7 1': 12345,'V6R 3M9': 12345,'M5T 2': 12345,'4221s':12345})

fec_2012["zip_valid"]=pd.DataFrame(fec_2012["zip_valid"].astype(float))
fec_2012["zip_valid"]=fec_2012["zip_valid"].fillna(value=0)

fec_2016["zip_valid"]=(fec_2016.contbr_zip).map(lambda x:str(x)[:5])
fec_2016["zip_valid"]=(fec_2016.contbr_zip).map({'NG7 1': 12345,'V6R 3M9': 12345,'M5T 2': 12345,'4221s':12345})

fec_2016["zip_valid"]=pd.DataFrame(fec_2016["zip_valid"].astype(float))
fec_2016["zip_valid"]=fec_2016["zip_valid"].fillna(value=0)


#Random forest treat int values as categorial


#using occupation as a feature

temp_fec = pd.DataFrame({'contbr_occupation': fec.contbr_occupation.unique(), 'contbr_occupation_id':range(len(fec.contbr_occupation.unique()))})
fec = fec.merge(temp_fec, on='contbr_occupation', how='left')


# Adding occupation id column for occupation names 2012

temp_fec_2012 = pd.DataFrame({'contbr_occupation': fec_2012.contbr_occupation.unique(), 'contbr_occupation_id':range(len(fec_2012.contbr_occupation.unique()))})
fec_2012 = fec_2012.merge(temp_fec_2012, on='contbr_occupation', how='left')



# Adding occupation id column for occupation names 2016

temp_fec_2016 = pd.DataFrame({'contbr_occupation': fec_2016.contbr_occupation.unique(), 'contbr_occupation_id':range(len(fec_2016.contbr_occupation.unique()))})
fec_2016 = fec_2016.merge(temp_fec_2016, on='contbr_occupation', how='left')


TrainingFeatures = fec[['contb_receipt_amt','zip_valid','contbr_occupation_id']]
TrainingClassLabels = fec['party']
classifier = RandomForestClassifier(n_estimators=100)
clf = classifier.fit(TrainingFeatures,TrainingClassLabels)

print classifier.max_features
#Testing on 2012 Data

TestFeatures = fec_2012[['contb_receipt_amt','zip_valid','contbr_occupation_id']]
TestClassLabels=fec_2012['party']
Test_predict=clf.predict(TestFeatures)

#Cross Validating on 2012 data

ValidationFeatures = fec_2012[['contb_receipt_amt','zip_valid','contbr_occupation_id']]
ValidationClassLabels=fec_2012['party']

scores=cross_validation.cross_val_score(clf,ValidationFeatures,ValidationClassLabels,cv=5)
print 'Scores'
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#K fold cross validation
labels = fec_2012.party

skf=StratifiedKFold(labels,3)
for train,test in skf:
    print("%s %s"%(train,test))


#Now we will predict 2016 results

PredictFeatures = fec_2016[['contb_receipt_amt','zip_valid','contbr_occupation_id']]
#TestClassLabels=fec_2012['party']
Predict2016 = clf.predict(PredictFeatures)

print len(Predict2016)
fec_2016["predict_winner"] = Predict2016

#For each state, predict winner of popular vote.

#for i in range(len())


print '2016 results'


Final_Democrat=0
Final_Republican=0
state = fec_2016.contbr_st.unique()
for j in range(len(state)):
    for i in range(len(fec_2016["predict_winner"])) :
        if fec_2016.contbr_st[i]==state[j]:
            Democrat=0
            Republican=0
            Other=0
            if fec_2016.predict_winner[i]=='Democratic': #and fec_2016.contbr_st[i]==state[j]:
                Democrat=Democrat+1
            elif  fec_2016.predict_winner[i]=='Republican': #and fec_2016.contbr_st[i]==state[j]:
                Republican=Republican+1
            else:
                Other=Other+1
    if Democrat>Republican:

        print 'Democrat Wins in 2016 as per the Predictive Model for state'
        print state[j]
        Final_Democrat=Final_Democrat+1
    else:
        print 'Republican Wins  in 2016 as per the Predictive Model for state'
        print state[j]
        Final_Republican=Final_Republican+1


print 'Democrat wins in states'
print Final_Democrat

print 'Republican wins in states'
print Final_Republican







cm=confusion_matrix(TestClassLabels,Test_predict)
#print cm
np.set_printoptions(precision=2)
print('Confusion matrix ,without normalization')
print(cm)
print 'Accuracy'
print accuracy_score(TestClassLabels, Test_predict, normalize=False)

labels=['Democratic','Republicans']
#plt.matshow(cm)
#plt.title('Normalized Confusion matrix')
#plt.colorbar()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Normalized Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#print clf.feature_importances_

cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
print('Normalized Confusion Matrix')
print(cm_normalized)
print 'Normalized Accuracy'
print accuracy_score(TestClassLabels, Test_predict)
labels=['Democratic','Republicans']
#plt.matshow(cm)
#plt.title('Normalized Confusion matrix')
#plt.colorbar()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm_normalized)
plt.title('Normalized Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
target_names = ['class 0', 'class 1','class 2']

print(classification_report(TestClassLabels, Test_predict, target_names=target_names))














