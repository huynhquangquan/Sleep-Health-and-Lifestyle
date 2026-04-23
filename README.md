# Sleep Health and Lifestyle Analysis

Dự án phân tích dữ liệu về sức khỏe giấc ngủ và lối sống, sử dụng dataset từ Kaggle để xây dựng mô hình dự đoán các rối loạn giấc ngủ như mất ngủ (Insomnia) và ngưng thở khi ngủ (Sleep Apnea).

## Tính năng

- Phân tích dữ liệu khám phá (EDA) trong notebook Jupyter
- Mô hình học máy Random Forest để dự đoán rối loạn giấc ngủ
- Ứng dụng web Flask để nhập dữ liệu và nhận dự đoán cùng lời khuyên

## Cấu trúc dự án

- `app.py`: Ứng dụng Flask API cho dự đoán
- `Health_and_Sleep_analysis.ipynb`: Notebook phân tích dữ liệu và xây dựng mô hình
- `predictcheck.py`: Script kiểm tra dự đoán
- `templates/`: Thư mục chứa template HTML
- `Sleep_health_and_lifestyle_dataset.csv`: Dataset gốc
- `processed_test.csv`, `results.csv`: Dữ liệu xử lý và kết quả
- `rf_model.pkl`, `label_encoder.pkl`, `onehot_encoder.pkl`: Mô hình và encoders đã huấn luyện

## Yêu cầu hệ thống

- Python 3.7+
- Flask
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib

## Cách chạy

1. Cài đặt các thư viện cần thiết:
   ```
   pip install flask pandas numpy scikit-learn matplotlib seaborn joblib
   ```

2. Chạy ứng dụng web:
   ```
   python app.py
   ```

3. Mở trình duyệt và truy cập `http://localhost:5000`

4. Để chạy notebook phân tích:
   ```
   jupyter notebook Health_and_Sleep_analysis.ipynb
   ```

## Nguồn dữ liệu

Dataset gốc từ: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
