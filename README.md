# Temporal NER with BERT

Hệ thống nhận diện thực thể (NER) trích xuất thông tin thời gian và sự kiện y khoa (Temporal Information Extraction) từ văn bản lâm sàng, sử dụng mô hình BERT fine-tuned cho bài toán Token Classification.

## 📋 Giới thiệu

Dự án xây dựng pipeline NER nhận diện 2 loại thực thể trong câu văn bản y khoa:
- **EVENT**: Sự kiện/triệu chứng y khoa (VD: `fever`, `chest pain`, `headache`)
- **TIME**: Biểu thức thời gian (VD: `5 days ago`, `yesterday`, `last week`)

Mô hình được so sánh với baseline SVM (dựa trên TF-IDF) để đánh giá hiệu quả của BERT trong bài toán trích xuất thông tin thời gian từ hồ sơ bệnh án.

## 🗂️ Dữ liệu

- Dataset: `temporal_dataset_300_advanced.csv` — 300 câu văn bản y khoa được gắn nhãn theo định dạng BIO
- Nhãn: `B-EVENT`, `B-TIME`, `I-TIME`, `O`
- Chia tập: Train 210 / Validation 45 / Test 45 (tỷ lệ 70/15/15)

## 🧠 Mô hình

- **Backbone**: `bert-base-uncased` (Hugging Face Transformers)
- **Kiến trúc**: `BertForTokenClassification`
- **Baseline so sánh**: SVM (kernel linear) trên đặc trưng TF-IDF

### Tham số huấn luyện
| Tham số | Giá trị |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 8 |
| Max sequence length | 128 |
| Weight decay | 0.01 |

## 📊 Kết quả

| Model | Precision | Recall | F1-score |
|---|---|---|---|
| BERT (fine-tuned) | 1.0000 | 1.0000 | 1.0000 |
| SVM (baseline) | 1.0000 | 1.0000 | 1.0000 |

> ⚠️ **Lưu ý**: F1 = 1.0 trên cả hai mô hình cho thấy dataset (300 mẫu, sinh tổng hợp) khá đơn giản/dễ phân biệt. Kết quả này có thể chưa phản ánh đúng hiệu năng thực tế trên dữ liệu lâm sàng đa dạng hơn — cần kiểm tra thêm khả năng overfitting và mở rộng dataset để đánh giá khách quan hơn.

## 🚀 Cách sử dụng

### Cài đặt
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets seqeval pandas numpy scikit-learn matplotlib
```

### Chạy inference
```python
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model_path = "path/to/temporal_ner_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

result = nlp("The patient developed fever 5 days ago and cough yesterday")
print(result)
```

## 📁 Cấu trúc project

```
temporal-ner-bert/
├── TEMPORAL_NER_WITH_BERT.ipynb   # Notebook chính: xử lý dữ liệu, train, đánh giá
├── temporal_dataset_300_advanced.csv  # Dataset gốc
└── README.md
```

## 🛠️ Công nghệ sử dụng

- Python, PyTorch, Hugging Face Transformers
- Datasets, Seqeval (đánh giá NER)
- Scikit-learn (baseline SVM)
- Pandas, NumPy, Matplotlib

## 👤 Tác giả

Tuấn Anh — Sinh viên ngành Trí tuệ nhân tạo, Đại học Công nghiệp TP.HCM (IUH)
