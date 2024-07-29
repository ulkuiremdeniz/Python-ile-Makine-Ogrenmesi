'''
 data preprocessing : veri ön işleme
 DataFrame : verinin tablo şeklinde gösterimine verilen ad(iki boyutlu veri)
 CSV : Comma-separated values verileri saklamak ve taşımak için kullanılan dosya formatı

 CRISP-DM
    -veri madenciliği projeleri için kulanılan bir süreç modelidir.
    -altı ana fazdan oluşur
 1.İş Anlayışı(Business Understanding)
    -proje gereksinimlerinin oluşturulması ve çizelgelerin oluşturulması.
 2.Veri Anlayışı(Data Understanding)
    -veri toplama ve analiz etme
 3.Veri Hazırlığı(Data Preparation)
    -veri temizleme,ön işleme,eksik verilerin tamamlanması,verinin uygun formata getirilmesi
 4.Modelleme(Modeling)
    -farklı modelleme tekniklerini seçme ve uygulama
 5.Değerlendirme (Evaluation)
    -modellerin performansını ve doğruluğunu değerlendirme,modelin amacına uygun olduğunu değerlendirme
 6.Dağıtım (Deployment)
    -modelin kullanıma sunulması ve sonuçların incelenmesi,dökümantasyon

'''



#KÜTÜPHANELER
'''Pandas:veri manipülasyonu ve analizi için kullanılır.
   veri okuma,veri yazma,veri temizleme...
   '''
import pandas as pd
'''NumPy:bilimsel hesaplamalarda kullanılır.'''
import numpy as np
'''Matplotlib:veri görselleştirme ve grafik çizimi için araçlar sağlar'''
import matplotlib.pyplot as plt



#VERİ ÖN İŞLEME
#1.VERİ YÜKLEME
veriler = pd.read_csv(r"C:\\Users\\PC\Desktop\\Python ile Makine Öğrenmesi\\veriler.csv")
print(veriler)



#2.EKSİK VERİLER
eksik_veriler = pd.read_csv(r"C:\\Users\\PC\Desktop\\Python ile Makine Öğrenmesi\\eksikveriler.csv")
print(eksik_veriler)

'''veri setindeki eksik verileri(missing values) işlemek için kullanılan bir sınıftır.'''
from sklearn.impute import SimpleImputer

'''eksik verileri(nan) diğer verilerin ortalaması alınıp bu değerle doldurmak için bir imputer nesnesi oluşturuluyor'''
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
'''iloc : DataFrame'deki satır ve sutunları konum tabanlı seçmek için kullanılır.(tüm satırları ve sutun değeri 1,2 ve 3 olan sutunları seçtik)'''
yas = eksik_veriler.iloc[:,1:4].values
print(yas)

#imputer nesnesini eksik değerleri nasıl dolduracağına dair eğittik.
imputer = imputer.fit(yas[:,1:4])
#hesaplanan ortalama değerler non değerlerin yerine yazılıyor
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)




'''Regresyon algoritmaları numeric verilerle çalışır.'''
#3.KATEGORİK VERİLER
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing
'''LabelEncoder : kategorik verileri sayısal değerlere dönüştürür'''
le = preprocessing.LabelEncoder()
'''ülkeler sayısal değerlere dönüştürülüyor'''
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)
'''OneHotEncoder : kategorik verileri ikili formata dönüştürür'''
ohe = preprocessing.OneHotEncoder()
#sayısal değerleri ikili bir vektöre dönüştürür
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)




#4.VERİ KÜMELERİNİN BİRLEŞTİRİLMESİ
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data =yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=cinsiyet ,index=range(22),columns=['cinsiyet'])

#axis birleştirme işleminin sutun bazında yapılacağını belirtir
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
s2 = pd.concat([s,sonuc3],axis=1)
print(s2)



'''Özellikler :ülke ,boy,kilo,yas bilgilerine bakılarak kişinin cinsiyet(hedef) tahmini yapılması amaçlanmaktadır.'''
#5.VERİLERİ KÜMESİNİN EĞİTİM VE TEST VERİLERİNE BÖLÜNMESİ
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(s,sonuc3,test_size=0.33,random_state=0)



#6.ÖZNİTELİK ÖLÇEKLEME
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)





