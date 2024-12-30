import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_excel(r'C:\Users\admin\Desktop\Veri Madencilği\kredi_analizi.xlsx')

label_encoder = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('LoanApproved', axis=1)
y = df['LoanApproved']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

feature_importance = pd.Series(rf.feature_importances_, index=X.columns)

top_features_rf = feature_importance.sort_values(ascending=False).head(10)

print("Top features selected by RandomForestClassifier:")
print(top_features_rf)

plt.figure(figsize=(10, 8))
sns.barplot(x=top_features_rf, y=top_features_rf.index)
plt.title('Top 10 Features Selected by RandomForestClassifier')
plt.xlabel('Feature Importance')
plt.show()
