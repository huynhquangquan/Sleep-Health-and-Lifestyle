from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load mô hình và các bộ chuyển đổi đã lưu
model = joblib.load("rf_model.pkl")
onehot_encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Các đặc trưng đầu vào từ request
numeric_features = [
    "Age", "Sleep Duration", "Physical Activity Level",
    "Systolic", "Diastolic", "Heart Rate", "Daily Steps"
]
categorical_features = ["Gender", "Occupation", "BMI Category"]


recommendations = {
    "None": "Bạn không có dấu hiệu rối loạn giấc ngủ. Hãy duy trì thói quen ngủ lành mạnh!",
    "Insomnia": "Bạn có dấu hiệu mất ngủ. Hãy thử duy trì lịch ngủ cố định, tránh caffeine vào buổi tối, hãy thư giãn, đừng suy nghĩ về chuyện không tốt đẹp trước khi ngủ. Nếu triệu chứng kéo dài, hãy gặp bác sĩ chuyên khoa giấc ngủ.",
    "Sleep Apnea": "Bạn có thể bị chứng ngưng thở khi ngủ. Hãy thử giảm cân nếu cần, tránh rượu trước khi ngủ và tham khảo ý kiến bác sĩ về CPAP hoặc các phương pháp điều trị khác.",
}

def validate_prediction(predicted_label, features):
    heart_rate = features.get("Heart Rate", None)
    sleep_duration = features.get("Sleep Duration", None)
    bmi_category = features.get("BMI Category", None)
    age = features.get("Age", None)
    physical_activity = features.get("Physical Activity Level", None)
    systolic = features.get("Systolic", None)
    diastolic = features.get("Diastolic", None)
    odd_symtom = False

    # Cơ bản theo nhãn dự đoán
    base_recommendation = recommendations[predicted_label]
    
    # Kiểm tra và bổ sung gợi ý dựa trên các đặc trưng
    if predicted_label == "Sleep Apnea":
        if heart_rate is not None and heart_rate < 40:
            base_recommendation += "Khó thở khi ngủ, nhịp tim đập chậm! Hãy đến bác sĩ chuyên khoa tim mạch để kiểm tra. "
        if bmi_category in ["Overweight", "Obese"]:
            base_recommendation += "Ngoài ra, BMI của bạn cao, hãy cân nhắc giảm cân để cải thiện tình trạng. "
        elif bmi_category == "Underweight":
            base_recommendation += "Ngoài ra, BMI của bạn thấp, hãy cân nhắc tăng cân để cải thiện tình trạng. "
        if age is not None and age > 50:
            base_recommendation += "Ở độ tuổi này, ngưng thở khi ngủ có thể nghiêm trọng hơn, hãy kiểm tra sức khỏe tổng quát. "

    elif predicted_label == "Insomnia":
        if sleep_duration is not None and sleep_duration < 5:
            base_recommendation += "Thời gian ngủ của bạn quá ngắn, hãy thử tạo không gian yên tĩnh và tránh màn hình trước khi ngủ. "
        if physical_activity is not None and physical_activity < 30:  # Giả sử < 30 là mức thấp
            base_recommendation += "Bạn nên thử tập thể dục nhẹ nhàng vào ban ngày để cải thiện chất lượng giấc ngủ. "
        if heart_rate is not None and heart_rate > 100:
            base_recommendation += "Nhịp tim của bạn cao, hãy thử thực hành hơi thở sâu để giảm căng thẳng. "

    elif predicted_label == "None":
        if sleep_duration is not None and sleep_duration < 6:
            base_recommendation += "Tuy nhiên, bạn ngủ hơi ít, hãy cố gắng ngủ đủ 7-8 tiếng mỗi đêm để duy trì sức khỏe. "

    if systolic is not None and diastolic is not None:
        if (90 > systolic or systolic > 180) or (60 > diastolic or diastolic > 140):
            base_recommendation += "Huyết áp của bạn không ổn định. "
            odd_symtom = True
    
    if heart_rate is not None:
        if 40 > heart_rate or heart_rate > 150:
            base_recommendation += "Nhịp tim của bạn không ổn định. "
            odd_symtom = True

    if odd_symtom:
        return f"{base_recommendation} Hãy tham khảo ý kiến bác sĩ để kiểm tra sức khỏe tổng quát. "
    # Nếu không có điều kiện đặc biệt nào khớp, trả về gợi ý cơ bản
    else: return base_recommendation

@app.route('/')
def home():
    return render_template('html.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        missing_fields = [field for field in numeric_features + categorical_features if field not in data]
        if missing_fields:
            return jsonify({"error": f"Thiếu các trường: {missing_fields}"}), 400

        df_raw = pd.DataFrame([data])
        df_cat = df_raw[categorical_features]
        encoded_array = onehot_encoder.transform(df_cat)
        df_encoded = pd.DataFrame(
            encoded_array,
            columns=onehot_encoder.get_feature_names_out(categorical_features)
        )

        df_numeric = df_raw[numeric_features]
        df_final = pd.concat([df_numeric.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

        expected_features = model.feature_names_in_.tolist()
        df_final = df_final.reindex(columns=expected_features, fill_value=0)

        prediction = model.predict(df_final)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        recommendation = validate_prediction(predicted_label, data)

        return jsonify({"prediction": predicted_label, "advice": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
