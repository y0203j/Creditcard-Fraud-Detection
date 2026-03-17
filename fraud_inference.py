import pandas as pd
import joblib

class AUSTRAC_FraudInference:
    def __init__(self, model_path, scaler_path):
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        self.features = self.model.get_booster().feature_names

    def process_and_predict(self, raw_data, threshold=0.1):
        """
        raw_data: DataFrame with 30 columns (Time, V1-V28, Amount)
        threshold: The surgical 0.1 threshold we determined
        """
        data = raw_data.copy()
        
        data[['Time', 'Amount']] = self.scaler.transform(data[['Time', 'Amount']])
        
        data = data[self.features]
        
        probs = self.model.predict_proba(data)[:, 1]
        
        predictions = (probs >= threshold).astype(int)
        
        return pd.DataFrame({
            'Fraud_Probability': probs,
            'Detection_Flag': predictions,
            'Action_Required': ['Manual Review' if x == 1 else 'Auto-Pass' for x in predictions]
        })


pipeline = AUSTRAC_FraudInference('models/xgb_frauddetection_model.pkl', 'models/robust_scaler_timeamount.pkl')

sample_data = X_test_full.head(5)
results = pipeline.process_and_predict(sample_data)
print(results)