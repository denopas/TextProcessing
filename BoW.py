import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_colwidth = 8000

# TR: Örnek Türkçe dokümanlar 
# EN: Sample documents in Turkish
docs = ['Açýklama projenin ortaklarýndan Rus enerji devi Gazprom dan geldi. Yýllýk 63 milyar metreküp enerji',
        'ilk günündeki 20 yarýþ heyecanlýydý, 109 puan toplayan Türkiye, 12 ülke arasýnda 9. oldu ve yarýþ tamamlandý',
        'Cortananýn yeni iþletim sistemi Windows 10 un önemli bir parçasý olduðunu belirten Microsoft ; Google Android ve iOS cihazlarýndaki Dijital',
        'Teknoloji devi Google, Android in MMM sürümüyle birlikte bir çok sistemsel hatasýnýn düzeltileceðini',
        'Siroz hastalýðý ile ilgili detaylara dikkat çekerek, saðlýklý bir karaciðere sahip olmak hastalýk için',
        'Hastalýk çoðu kez yýllarca doðru taný konmamasý veya ciddiye alýnmamasý sebebi ile kýsýrlaþtýrýcý etki yapabiliyor, kronik aðrý,',
        'Ýlk 4 etaptan galibiyetle ayrýlan 18 yaþýndaki Razgatlýoðlu, Ýtalya daki yarýþta 3. sýrayý alarak ',
        'Helal gýda pazarý sanki 860 milyar dolarýn üzerinde'    
]
# TR: Dokümanlara ait sýnýflar 
# EN: Classes of documents
classes = ['ekonomi', 'spor', 'teknoloji', 'teknoloji', 'saglik', 'saglik', 'spor', 'ekonomi']

# TR: Özel karakterlerin dönüþümü 
# EN: Conversion of special Turkish characters to Latin forms
coding = {'ç': 'c', 'ý': 'i', 'ü': 'u', 'þ': 's', 'ð': 'g', 'ö': 'o', 'Ý': 'I' }
for i in range(len(docs)):
    for k, v in coding.items():
        docs[i] = docs[i].replace(k, v)

docs = np.array(docs)
df_docs = pd.DataFrame({'Dokuman': docs, 
                        'Sinif': classes})
df_docs = df_docs[['Dokuman', 'Sinif']]
#print (df_docs)

WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')

def norm_doc(single_doc):
    # TR: Dokümandan özel karakterleri ve sayýlarý at
    # EN: Remove special characters and numbers
    single_doc = re.sub(r'[^a-zA-Z\s]', '', single_doc, re.I|re.A)
    # TR: Dokümaný küçük harflere çevir
    # EN: Convert document to lowercase
    single_doc = single_doc.lower()
    single_doc = single_doc.strip()
    # TR: Dokümaný token'larýna ayýr
    # EN: Tokenize documents
    tokens = WPT.tokenize(single_doc)
    # TR: Stop-word listesindeki kelimeler hariç al
    # EN: Filter out the stop-words 
    filtered_tokens = [token for token in tokens if token not in stop_word_list]
    # TR: Dokümaný tekrar oluþtur
    # EN: Reconstruct the document
    single_doc = ' '.join(filtered_tokens)
    return single_doc

norm_docs = np.vectorize(norm_doc) #like magic :)
normalized_documents = norm_docs(docs)
#print(normalized_documents)


# TR: 1.Terim Sayma Adýmlarý
# EN: 1.Term Counting Steps
from sklearn.feature_extraction.text import CountVectorizer
BoW_Vector = CountVectorizer(min_df = 0., max_df = 1.)
BoW_Matrix = BoW_Vector.fit_transform(normalized_documents)
print (BoW_Matrix)

# TR: BoW_Vector içerisindeki tüm öznitelikleri al
# EN: Fetch al features in BoW_Vector
features = BoW_Vector.get_feature_names()
print ("features[50]:" + features[50])
print ("features[52]:" +features[52])

BoW_Matrix = BoW_Matrix.toarray()
print(BoW_Matrix)
# TR: Doküman - öznitelik matrisini göster
# EN: Print document by term matrice
BoW_df = pd.DataFrame(BoW_Matrix, columns = features)
#print(BoW_df)
#print(BoW_df.info())



# TR: 2.TFxIdf Hesaplama Adýmlarý
# EN: 2.TFxIdf Calculation Steps
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf_Vector = TfidfVectorizer(min_df = 0., max_df = 1., use_idf = True)
Tfidf_Matrix = Tfidf_Vector.fit_transform(normalized_documents)
Tfidf_Matrix = Tfidf_Matrix.toarray()
print(np.round(Tfidf_Matrix, 3))
# TR: Tfidf_Vector içerisindeki tüm öznitelikleri al
# EN: Fetch al features in Tfidf_Vector
features = Tfidf_Vector.get_feature_names()
# TR: Doküman - öznitelik matrisini göster
# EN: Print document by term matrice
print(pd.DataFrame(np.round(Tfidf_Matrix, 3), columns = features))