import pandas as pd


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            # Dosyadan veri yükle
            self.data = pd.read_excel(self.file_path)
            # self.data = pd.read_excel(self.file_path).head(100)
            print("Veri başarıyla yüklendi.")
        except Exception as e:
            print(f"Veri yüklenirken bir hata oluştu: {e}")
        return self.data
