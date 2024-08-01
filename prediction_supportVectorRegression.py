#SUPPORT VECTOR REGRESSION
'''
SUPPORT VECTOR REGRESSION(DESTEK VEKTÖR REGRESYONU)
Bir margin aralığında maximum veri noktasını bulmayı amaçlar.Çizilen doğrular,eğriler üzerinde min margin aralığında max veriyi elde etmek.
Sınıflandırmada  sınıfları birbirinden ayırmak için kullanılır.Hangi doğru sınıfları daha iyi ayırır (max margin'i hangi doğru sağlıyor) ?
'''

#KÜTÜPHANELER
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#VERİ ÖN İŞLEME
dataset = pd.read_csv(r'C:\Users\PC\Desktop\Python ile Makine Öğrenmesi\maaslar.csv')
print(dataset)

#data frame dilimleme (slice)
egitim_seviyesi = dataset.iloc[:,1:2]
maas = dataset.iloc[:,2:]

#NumPy array dönüşümü
x = egitim_seviyesi.values
y = maas.values


#Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.title('SVR Regression')
plt.scatter(x_olcekli, y_olcekli, color='red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='blue')
plt.show()