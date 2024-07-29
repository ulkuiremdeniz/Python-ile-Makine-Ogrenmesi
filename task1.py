'''
ONE-HOT ENCODING :
    -kategorik verileri sayısal verilere çevirmek için kullanılır.
    -her kategori için ayrı bir sutun oluşturur ve sadece ilgili kategorinin bulunduğu satıra 1 diğerlerine 0 değerini verir.
    -binary vektörlere dönüştürür
    -kategoriler arasında ilişki kurmaz
    -kategoriler arttıkça veri seti boyutu artar
    -sıralı olmayan kategoriler(nominal)
LABEL ENCODING:
    -kategorik verileri sayısal verilere çevirmek için kullanılır.
    -kategorik verileri doğrudan sayısal verilere çevirir.Her kategoriye bir tam sayı değeri atanır.
    -tam sayılara dönüştürür.
    -kategoriler arasında sıralı ilişki kurar
    -veri seti boyutunu arttırmaz
    -sıralı kategoriler(ordinal)
'''


#1.DATA PREPROCESSING
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yükleme
dataset = pd.read_csv(r"C:\Users\PC\Desktop\Python ile Makine Öğrenmesi\task1.csv")
print(dataset)

#verileri encode etmek (nümerikleştirmek)
from sklearn import preprocessing
#tüm sutunlarda encode işlemi uygulanıyor
veriler = dataset.apply(preprocessing.LabelEncoder().fit_transform)
print(veriler)

#ilk sutunu one hot encoder ile kodluyoruz
outlook = veriler.iloc[:,:1]
print(outlook)

from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
outlook =ohe.fit_transform(outlook).toarray()
print(outlook)

#verilerin birleştirilerek dataframe elde edilmesi
sonuc1 =pd.DataFrame(data=outlook,index=range(14),columns=['Overcast','Rainy','Sunny'])
temperature_humidity = dataset.iloc[:,1:3]
sonuc3 =pd.DataFrame(data=veriler.iloc[:,-2:].values,index=range(14),columns=['Windy','Play'])

s = pd.concat([sonuc1,sonuc3],axis=1)
s2 = pd.concat([s,temperature_humidity],axis=1)
print(s2)


#bağımlı değişken :humidity
#VERİLERİN EĞİTİM VE TEST VERİ SETİNE BÖLÜNMESİ
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s2.iloc[:,:-1],s2.iloc[:,-1:],test_size=0.33,random_state=0)

#MODEL EĞİTİMİ
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)


#GERİYE DOĞRU ELEME (BACKWARD ELIMINATION)
import statsmodels.api as sm
x =np.append(arr = np.ones((14,1)).astype(int),values=s2.iloc[:,:-1],axis=1)
x_list = s2.iloc[:,[0,1,2,3,4,5]].values
x_list = np.array(x_list,dtype=float)
model = sm.OLS(s2.iloc[:,-1:],x_list).fit()
print(model.summary())

#veri elendi
x_list = s2.iloc[:,[0,1,2,4,5]].values
x_list = np.array(x_list,dtype=float)
model = sm.OLS(s2.iloc[:,-1:],x_list).fit()
print(model.summary())


#MODELİN TEKRAR EĞİTİLMESİ
#windy kolonunu kaldırıp modeli tekrar eğitiliyor

result =pd.concat([s2.iloc[:,0:3],s2.iloc[:,4:6]],axis=1)
print(result)

x_train, x_test, y_train, y_test = train_test_split(result,s2.iloc[:,-1:],test_size=0.33,random_state=0)
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

