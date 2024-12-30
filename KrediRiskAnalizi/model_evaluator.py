import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, results, X_test_scaled, y_test):
        self.results = results
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test

    def evaluate(self):
        """Evaluate models, compute performance metrics, and save them to an Excel file"""
        evaluation_data = []  # All results will be stored in this list.

        try:
            # Ensure the "results" directory exists
            output_dir = 'results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Compute metrics for each model
            for result in self.results:
                model = result['Trained Model']  # Get the trained model
                y_pred = model.predict(self.X_test_scaled)

                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)

                # Calculate confusion matrix and convert to percentages
                conf_matrix = confusion_matrix(self.y_test, y_pred)
                conf_matrix_percentage = (conf_matrix / conf_matrix.sum()) * 100  # Convert matrix to percentages

                # Add evaluation data (append result to the list)
                evaluation_data.append({
                    "Model": result['Model'],  # Model Name
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Confusion Matrix (Percentage)": str(conf_matrix_percentage.tolist())
                    # Convert matrix to string for easy viewing
                })

            # Save results to an Excel file in the "results" folder
            df = pd.DataFrame(evaluation_data)
            df.to_excel(os.path.join(output_dir, 'model_evaluation_results.xlsx'), index=False)
            print("Results successfully saved: results/model_evaluation_results.xlsx")

        except Exception as e:
            print(f"An error occurred during model evaluation: {e}")


# import os
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, roc_auc_score
#
#
# class ModelEvaluator:
#     def __init__(self, results, X_test_scaled, y_test):
#         self.results = results
#         self.X_test_scaled = X_test_scaled
#         self.y_test = y_test
#
#     def evaluate(self):
#         """Modelleri değerlendirin, performans ölçümlerini hesaplayın ve bunları Excel dosyasına kaydedin"""
#         evaluation_data = []  # Tüm sonuçlar bu listeye kaydedilecektir.
#
#         try:
#             # "Sonuçlar" klasörünün olduğundan emin olun.
#             output_dir = 'results'
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#
#             plt.figure(figsize=(15, 10))  # Şekil boyut ayarı
#
#             # Her model için boyutları hesaplayın
#             for result in self.results:
#                 model = result['Trained Model']  # Eğitilmiş modeli al
#                 y_pred = model.predict(self.X_test_scaled)
#                 y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
#
#                 accuracy = accuracy_score(self.y_test, y_pred)
#                 precision = precision_score(self.y_test, y_pred)
#                 recall = recall_score(self.y_test, y_pred)
#                 f1 = f1_score(self.y_test, y_pred)
#                 auc_score = roc_auc_score(self.y_test, y_proba)
#
#                 # حساب مصفوفة الالتباس وتحويلها إلى نسب مئوية
#                 conf_matrix = confusion_matrix(self.y_test, y_pred)
#                 conf_matrix_percentage = (conf_matrix / conf_matrix.sum()) * 100  # تحويل المصفوفة إلى نسب مئوية
#
#                 # Değerlendirmeye veri ekleyin (sonucu listeye kaydedin)
#                 evaluation_data.append({
#                     "Model": result['Model'],  # Model Adı
#                     "Accuracy": accuracy,
#                     "Precision": precision,
#                     "Recall": recall,
#                     "F1 Score": f1,
#                     "ROC AUC": auc_score,
#                     "Confusion Matrix (Percentage)": str(conf_matrix_percentage.tolist())
#                     # Matrisi kolay ezberleme için metne dönüştürün
#                 })
#
#                 # Her model için ROC eğrisini çizin.
#                 fpr, tpr, _ = roc_curve(self.y_test, y_proba)
#                 plt.plot(fpr, tpr, label=f"{result['Model']} (AUC = {auc_score:.4f})")
#
#             # ROC eğrisi grafiğinin hazırlanması
#             plt.plot([0, 1], [0, 1], 'k--')
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('ROC Curve for Different Models')
#             plt.legend(loc="lower right")
#
#             # Tabloyu "Sonuçlar" klasörüne resim olarak kaydedin.
#             plt.savefig(os.path.join(output_dir,
#                                      'roc_curve.png'))  # Resmi "Sonuçlar" klasörünün içine PNG dosyası olarak kaydedin.
#             print("ROC Curve başarıyla kaydedildi: results/roc_curve.png")
#             plt.show()  # Tabloyu görüntüle
#
#             # Sonuçları "Sonuçlar" klasörünün içindeki bir Excel dosyasına kaydedin.
#             df = pd.DataFrame(evaluation_data)
#             df.to_excel(os.path.join(output_dir, 'model_evaluation_results.xlsx'),
#                         index=False)  # Sonuçları Excel dosyasına kaydet
#             print("Sonuçlar başarıyla kaydedildi: results/model_evaluation_results.xlsx")
#
#         except Exception as e:
#             print(f"Modeller değerlendirilirken hata oluştu: {e}")
