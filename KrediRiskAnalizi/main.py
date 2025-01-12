from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


#Verileri yukleme
data_loader = DataLoader(r'C:\Users\admin\Desktop\Veri Madencilği\kredi_risk_analizi.xlsx')
data = data_loader.load_data()

# Veri hazırlama
data_preprocessor = DataPreprocessor(data, 'Credit Approval')
data_preprocessor.preprocess()

# Modellerin tanımı
models = {
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Model eğitimi
model_trainer = ModelTrainer(models)
model_trainer.train_models(data_preprocessor.X_train_scaled, data_preprocessor.y_train)

# Model Değerlendirmesi
model_evaluator = ModelEvaluator(model_trainer.results, data_preprocessor.X_test_scaled, data_preprocessor.y_test)
model_evaluator.evaluate()
