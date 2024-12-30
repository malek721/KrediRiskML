import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        # Verilerin bir DataFrame olduğundan emin olun
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Veri seti Series tipi değil, DataFrame olmalı.")

        try:
            # Giriş X'i ve hedef y'yi ayıkla
            X = self.data.drop(self.target_col, axis=1)
            y = self.data[self.target_col]

            # Verileri eğitim ve test olarak ayırma
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standart ölçüm uygulaması
            scaler = StandardScaler()
            self.X_train_scaled = scaler.fit_transform(X_train)
            self.X_test_scaled = scaler.transform(X_test)
            self.y_train = y_train
            self.y_test = y_test
            print("Veri başarıyla işlendi.")
        except Exception as e:
            print(f"Veri işlenirken hata oluştu: {e}")
