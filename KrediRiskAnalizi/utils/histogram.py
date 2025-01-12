# Gerekli kütüphaneleri içe aktarma
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Excel dosyasından veri yükle
data = pd.read_excel(r'C:\Users\admin\Desktop\Veri Madencilği\kredi_risk_analizi.xlsx')

# Histogram çizmek istediğiniz özelliklerin listesi.
features = ['Credit Score', 'Annual Income', 'InterestRate']

# Griddeki satır ve sütun sayısını belirtin.
num_features = len(features)
cols = 2  # Gridde olmasını istediğiniz sütun sayısı (2 veya 3 olarak ayarlayabilirsiniz)
rows = math.ceil(num_features / cols)

# Şekil ve ızgara oluştur
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))

# Boyutlar 1'den büyükse ızgarayı listeye düzleştir
axes = axes.flatten()

# Her özellik için histogram çizin.
for i, feature in enumerate(features):
    sns.histplot(data[feature].dropna(), bins=30, kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {feature}', fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].grid(axis='y', alpha=0.75)

# Grafik sayısı ızgara sayısından azsa boş eksenleri gizle
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
