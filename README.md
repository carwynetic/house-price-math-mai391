# 🏡 House Price Prediction using PCA & Linear Regression (From Scratch)

**Course:** MAI391 - Mathematics for Machine Learning  
**University:** FPT University  

## 📌 Giới thiệu dự án
Dự án này xây dựng mô hình dự đoán giá nhà (dựa trên tập dữ liệu Ames Housing) hoàn toàn từ con số 0. Thay vì sử dụng các thư viện Machine Learning đóng gói sẵn (như `scikit-learn`), chúng tôi tự cài đặt các thuật toán bằng `numpy` để chứng minh sự hiểu biết sâu sắc về **Đại số tuyến tính** và **Giải tích nhiều biến**.

Dự án áp dụng quy trình 2 bước:
1. **Dimensionality Reduction (PCA):** Nén các đặc trưng đa chiều xuống không gian 2 chiều để loại bỏ nhiễu và đa cộng tuyến.
2. **Linear Regression:** Tối ưu hóa hàm mất mát (MSE) để tìm ra bộ trọng số dự đoán giá nhà chuẩn xác nhất.

## 🧮 Nền tảng Toán học (Mathematical Foundations)

Dự án lập trình trực tiếp các công thức toán học sau thành mã nguồn:

### 1. Principal Component Analysis (PCA)
* **Ma trận Hiệp phương sai (Covariance Matrix):** $$S = \frac{1}{N} \mathbf{X}^T \mathbf{X}$$
  *(trong đó **X** là ma trận dữ liệu đã được chuẩn hóa mean-centered)*

* **Phân tích Trị riêng (Eigendecomposition):** Tìm Trị riêng $\lambda$ và Vector riêng $\mathbf{v}$ từ phương trình:
  $$S \mathbf{v} = \lambda \mathbf{v}$$

### 2. Linear Regression & Hàm mất mát (MSE Loss)
* **Hàm mục tiêu:** $L(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$

### 3. Phương pháp Tối ưu hóa (Optimization)
Mô hình đối chiếu 2 phương pháp để tìm vector trọng số $\mathbf{w}$:
* **Nghiệm giải tích (Normal Equation):** Chạm đáy hàm Loss bằng một phép tính duy nhất thông qua nghịch đảo ma trận.
  $$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
* **Nghiệm xấp xỉ (Gradient Descent):** Tối ưu hóa liên tục bằng vòng lặp ngược hướng đạo hàm (sử dụng Learning Rate $\alpha = 0.5$, 2000 epochs).
  $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$$

## 📂 Cấu trúc Repository

```text
house-price-math-mai391/
│
├── data/
│   └── train.csv                 # Tập dữ liệu gốc (Ames Housing)
│
├── notebooks/
│   └── Linear_Regression.ipynb   # Mã nguồn chính (Cài đặt thuật toán bằng Numpy)
│
└── README.md                     # Tài liệu mô tả dự án
