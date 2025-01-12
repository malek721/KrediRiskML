class ModelTrainer:
    def __init__(self, models):
        self.models = models
        self.results = []  # To store model training results

    def train_models(self, X_train, y_train):
        """""Modelleri eğit ve sonuçları kaydet"""""
        try:
            for model_name, model in self.models.items():
                """ Modelleri eğit """
                model.fit(X_train, y_train)

                # Eğitilmiş modeli self.results'a ekleyin
                self.results.append({
                    'Model': model_name,
                    'Trained Model': model
                })

            print("Modeller başarıyla eğitildi.")
        except Exception as e:
            print(f"Modeller eğitilirken hata oluştu: {e}")
