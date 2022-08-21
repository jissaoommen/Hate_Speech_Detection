import os
import pandas
import numpy
import nltk
import re
import matplotlib.pyplot as plt
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tfidf import feature_extraction
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay,classification_report,roc_curve,precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
path=os.getcwd()

data_path=os.path.join(path,"labeled_data.csv")

data=pandas.read_csv(data_path)
print(data)
y=[]

shortword=re.compile(r'\W*\b\w{1,3}\b')
stop_words=set(stopwords.words('english'))
print(stop_words)
lemmatizer=WordNetLemmatizer()
ps=PorterStemmer()
doc=[]
for i in range(0,3000):
    twt=data["tweet"][i]
    # print(twt)
    twt=re.sub(r'[\W]',' ',twt)
    twt=re.sub('\d+','',twt)
    twt=twt.lower()
    twt=shortword.sub('',twt)
    twt=word_tokenize(twt)
    twt=[word for word in twt if not word in stop_words]
    twt=[ps.stem(word) for word in twt]
    twt=' '.join(twt)
    doc.append(twt)
    # print(twt)
    classes=data["class"][i]
    
    if classes==0 or classes==1 :
        y.append(0)
    elif classes==2: 
        y.append(1)

y=numpy.array(y)
x=feature_extraction(doc)[:,0:1500]
print(Counter(y))
over_sample=SMOTE()
x,y=over_sample.fit_resample(x,y)
print(Counter(y))
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)
print(x.shape)
print(xtrain.shape)
print(xtest.shape)

svm_model=SVC(probability=True)
svm_model.fit(xtrain,ytrain)
prediction=svm_model.predict(xtest)
print(prediction)
print(ytest)
count=0
for i in range(0,len(ytest)):
    if ytest[i]==prediction[i]:
        count=count+1
print(count)
accuracy=(count/len(ytest))*100
print('accuracy:',accuracy)
tp=0
tn=0
fp=0
fn=0
for i in range(0,len(ytest)):
    if ytest[i]==0 and prediction[i]==0:
        tp=tp+1
    elif ytest[i]==0 and prediction[i]==1:
        fp=fp+1
    elif ytest[i]==1 and prediction[i]==0:
        fn=fn+1
    else:

        tn=tn+1
print('true positive:',tp)
print('false positive:',fp)
print('false negative:',fn)
print('true negative:',tn)
cm=[[tp,fp],[fn,tn]]
cm=numpy.array(cm) 
d=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['hate speech','clean speech'])  
d.plot()
plt.title('svm')
plt.show()
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f1_score=2*(recall*precision)/(precision+recall)
print('precision:',precision)
print('recall:',recall)
print('f1score:',f1_score)
prob=svm_model.predict_proba(xtest)
prob=prob[:,1]
result_fpr1,result_tpr1,_=roc_curve(ytest,prob)
result_precision1,result_recall1,_=precision_recall_curve(ytest,prob)
# plt.xlabel('false positive rate')
# plt.ylabel('true positive rate')
# plt.legend()
# plt.show()

rf_model=RandomForestClassifier()
rf_model.fit(xtrain,ytrain)
prediction=rf_model.predict(xtest)
print(prediction)
print(ytest)
count=0
for i in range(0,len(ytest)):
    if ytest[i]==prediction[i]:
        count=count+1
print(count)
accuracy=(count/len(ytest))*100
print(accuracy)
tp=0
tn=0
fp=0
fn=0
for i in range(0,len(ytest)):
    if ytest[i]==0 and prediction[i]==0:
        tp=tp+1
    elif ytest[i]==0 and prediction[i]==1:
        fp=fp+1
    elif ytest[i]==1 and prediction[i]==0:
        fn=fn+1
    else:

        tn=tn+1
print('true positive:',tp)
print('false positive:',fp)
print('false negative:',fn)
print('true negative:',tn)
cm=[[tp,fp],[fn,tn]]
cm=numpy.array(cm) 
d=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['hate speech','clean speech'])  
d.plot()
plt.title('random forest')
plt.show()
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f1_score=2*(recall*precision)/(precision+recall)
print('precision:',precision)
print('recall:',recall)
print('f1score:',f1_score)
prob=rf_model.predict_proba(xtest)
prob=prob[:,1]
result_fpr2,result_tpr2,_=roc_curve(ytest,prob)
result_precision2,result_recall2,_=precision_recall_curve(ytest,prob)
# plt.xlabel('false positive rate')
# plt.ylabel('true positive rate')
# plt.legend()
# plt.show()

lg_model=LogisticRegression()
lg_model.fit(xtrain,ytrain)
prediction=lg_model.predict(xtest)
print(prediction)
print(ytest)
count=0
for i in range(0,len(ytest)):
    if ytest[i]==prediction[i]:
        count=count+1
print(count)
accuracy=(count/len(ytest))*100
print(accuracy)
tp=0
tn=0
fp=0
fn=0
for i in range(0,len(ytest)):
    if ytest[i]==0 and prediction[i]==0:
        tp=tp+1
    elif ytest[i]==0 and prediction[i]==1:
        fp=fp+1
    elif ytest[i]==1 and prediction[i]==0:
        fn=fn+1
    else:

        tn=tn+1
print('true positive:',tp)
print('false positive:',fp)
print('false negative:',fn)
print('true negative:',tn)
cm=[[tp,fp],[fn,tn]]
cm=numpy.array(cm) 
d=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['hate speech','clean speech'])  
d.plot()
plt.title('logistic regression')
plt.show()
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f1_score=2*(recall*precision)/(precision+recall)
print('precision:',precision)
print('recall:',recall)
print('f1score:',f1_score)
prob=lg_model.predict_proba(xtest)
prob=prob[:,1]
prob1=[0 for _ in range(len(ytest)) ]
s_fpr,s_tpr,_=roc_curve(ytest,prob1)
result_fpr,result_tpr,_=roc_curve(ytest,prob)
plt.plot(s_fpr,s_tpr,linestyle='--',label='no skill')
plt.plot(result_fpr1,result_tpr1,marker='.',label='svm')
plt.plot(result_fpr2,result_tpr2,marker='.',label='random forest')
plt.plot(result_fpr,result_tpr,marker='.',label='logistic regression')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('roc curve')
plt.legend()
plt.show()
result_precision,result_recall,_=precision_recall_curve(ytest,prob)
s_precision,s_recall,_=precision_recall_curve(ytest,prob1)
no_skill=len(ytest[ytest==1])/len(ytest)
plt.plot([0,1],[0,0],linestyle='--',label='no skill')
plt.plot(result_precision1,result_recall1,marker='.',label='svm')
plt.plot(result_precision2,result_recall2,marker='.',label='random forest')
plt.plot(result_precision,result_recall,marker='.',label='logistic regression')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision-recall curve')
plt.legend()
plt.show()
pickle.dump(svm_model,open('svm_model.pkl','wb'))
pickle.dump(rf_model,open('rf_model.pkl','wb'))
pickle.dump(lg_model,open('lg_model.pkl','wb'))
pickle.dump(doc,open('doc.pkl','wb'))