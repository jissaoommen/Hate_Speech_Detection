import warnings
warnings.filterwarnings('ignore')
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tfidf import feature_extraction

svm_model=pickle.load(open('svm_model.pkl','rb'))
rf_model=pickle.load(open('rf_model.pkl','rb'))
lg_model=pickle.load(open('lg_model.pkl','rb'))
doc=pickle.load(open('doc.pkl','rb'))

shortword=re.compile(r'\W*\b\w{1,3}\b')
stop_words=set(stopwords.words('english'))

lemmatizer=WordNetLemmatizer()
ps=PorterStemmer()


test_input=input('enter the tweet:')
# print(test_input)
twt=re.sub(r'[\W]',' ',test_input)
twt=re.sub('\d+','',twt)
twt=twt.lower()
twt=shortword.sub('',twt)
twt=word_tokenize(twt)
twt=[word for word in twt if not word in stop_words]
twt=[ps.stem(word) for word in twt]
twt=' '.join(twt)
doc.append(twt)
x=feature_extraction(doc)[:,0:1500]
m,n=x.shape
# print(m,n)
x=x[m-1]
x=x.reshape(1,n)
svm=svm_model.predict(x)
d={0:'hate speech',1:'clean speech'}
print('svm output:',d[svm[0]])
rf_model=rf_model.predict(x)
print('rf_model:',d[rf_model[0]])
lg_model=lg_model.predict(x)
print('lg_model:',d[lg_model[0]])