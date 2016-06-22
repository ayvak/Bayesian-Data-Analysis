from nltk import word_tokenize
from nltk.corpus import stopwords
t="Kevin is a good boy. Meeta is a bad girl. Varun is a horrible boy. Meenu is beautiful girl. Raju is a good boy. Mayank is good boy. Maria is good girl."
m=word_tokenize(t)
for word in m: # iterate over word_list
  if word in stopwords.words('english'): 
    m.remove(word)
  if len(word)==1:
      m.remove(word)
m=list(set(m))
dict={m[0]:'False',m[2]:'False',m[3]:'False',m[4]:'False',m[5]:'False',m[7]:'False',m[9]:'False',m[10]:'False',m[11]:'False',m[12]:'False',m[13]:'False',m[1]:'True',m[6]:'True',m[8]:'True'}
