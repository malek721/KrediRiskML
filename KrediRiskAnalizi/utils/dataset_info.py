import pandas as pd

# Veriyi yükleme
data = pd.read_excel(r'C:\Users\admin\Desktop\Veri Madencilği\kredi_risk_analizi.xlsx')

# Genel veri bilgilerini gösterme
print("Veri hakkında bilgiler:")
print(data.info())

# Satır sayısını (örnek sayısını) öğrenme
print(f"Toplam örnek sayısı: {data.shape[0]}")

# Verinin ilk 5 satırını gösterme
print("Verinin ilk 5 satırı:")
print(data.head())

# Her sütundaki benzersiz değerlerin sayısını öğrenme
print("Her sütundaki benzersiz değer sayısı:")
print(data.nunique())

# Her sütundaki eksik (NaN) değer sayısını öğrenme
print("Her sütundaki eksik değer sayısı:")
print(data.isnull().sum())

# Sayısal sütunlar için temel istatistiksel bilgiler
print("İstatistiksel özet (sayısal sütunlar):")
print(data.describe())

# Sütun adlarını listeleme
print("Verideki sütun adları:")
print(data.columns)

# Hesaplama: Her sütun için ortalama (mean) ve en sık tekrar eden değer (mode)
print("Her sütundaki ortalama değer (mean):")
print(data.mean())

print("Her sütundaki en sık tekrar eden değer (mode):")
print(data.mode().iloc[0])  # mode() döndürülen DataFrame'den ilk satırı alıyoruz çünkü bir sütun için birden fazla mod olabilir
