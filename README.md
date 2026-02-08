# JD-KAN: Jump-Diffusion Kolmogorov–Arnold Network

**JD-KAN** là mô hình dự báo chuỗi thời gian thế hệ mới, được thiết kế để xử lý đồng thời xu hướng diễn biến liên tục (Continuous Flow) và các cú sốc bất ngờ (Discrete Jumps). Mô hình lấy cảm hứng từ lý thuyết Jump–Diffusion trong tài chính và kiến trúc Kolmogorov–Arnold Networks (KAN), cho phép ngoại suy xu hướng mượt mà đồng thời phản ứng tức thì với các biến động mạnh.

---

## Nội dung
- [Giới thiệu](#giới-thiệu)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Các thành phần chính](#các-thành-phần-chính)
  - [1. Cascaded Frequency Decomposition (CFD)](#1-cascaded-frequency-decomposition-cfd)
  - [2. Branch 1: Continuous Flow Stream (Luồng Liên Tục)](#2-branch-1-continuous-flow-stream-luồng-liên-tục)
  - [3. Branch 2: Discrete Jump Stream (Luồng Nhảy Vọt)](#3-branch-2-discrete-jump-stream-luồng-nhảy-vọt)
  - [4. Adaptive Fusion Layer (Lớp Hợp Nhất)](#4-adaptive-fusion-layer-lớp-hợp-nhất)
- [Định nghĩa toán học](#định-nghĩa-toán-học)

---

## Giới thiệu
JD-KAN hướng đến việc mô hình hóa các chuỗi thời gian có hai thành phần cơ bản:

- **Dòng liên tục** (Continuous Flow): các biến đổi mượt mà, dài hạn (trend) và tuần hoàn (seasonality).
- **Cú nhảy rời rạc** (Discrete Jumps): các sự kiện bất thường, spikes hoặc drops, đòi hỏi cơ chế phát hiện và tái tạo riêng biệt.

Mục tiêu: kết hợp ưu điểm của các mô hình nội suy mượt (rKAN/Pade-like approximations) và các cơ chế nhớ/ghi nhận sự kiện để cải thiện khả năng dự báo trong môi trường có nhiễu và shock.

---

## Kiến trúc mô hình
Luồng xử lý dữ liệu của JD-KAN được chia thành 3 giai đoạn chính:

1. **Phân rã (Decomposition)**: tách tín hiệu thành nhiều dải tần (trend / seasonality / residual).
2. **Xử lý song song (Dual-Stream Processing)**: hai nhánh xử lý riêng biệt cho phần liên tục và phần nhảy.
3. **Hợp nhất (Adaptive Fusion)**: gộp kết quả bằng tham số học được để cho ra dự báo cuối cùng.

---

## Các Thành Phần Chính

### 1. Cascaded Frequency Decomposition (CFD)
Thay vì dùng một lớp Moving Average đơn lẻ, JD-KAN dùng cơ chế phân rã đa tầng để tách tín hiệu thành 3 thành phần:

- **Trend (Low-freq)**: xu hướng dài hạn, xử lý bởi kernel lớn.
- **Seasonality (Mid-freq)**: dao động tuần hoàn, xử lý bởi kernel trung bình.
- **Residual (High-freq)**: phần dư chứa nhiễu và các tín hiệu sốc (spikes/drops), là đầu vào quan trọng cho nhánh Jump.

---

### 2. Branch 1: Continuous Flow Stream (Luồng Liên Tục)
Nhánh này chịu trách nhiệm dự báo phần nền tảng ổn định của chuỗi thời gian.

- **Core Technology**: sử dụng **Rational KAN (rKAN)** với hàm kích hoạt là đa thức Jacobi \(P(x)/Q(x)\).
  - *Ưu điểm*: tránh hiện tượng cực điểm (poles) và nổ gradient thường thấy ở Pade approximation.

- **Adaptive Complexity**:
  - *Trend*: sử dụng rKAN bậc thấp (Order 2–3) để đảm bảo độ mượt.
  - *Season*: sử dụng rKAN bậc cao (Order 4–5) để bắt các mẫu phức tạp.

- **Linear Projection**: chiếu đặc trưng từ không gian quá khứ sang tương lai (feature projection).

---

### 3. Branch 2: Discrete Jump Stream (Luồng Nhảy Vọt)
Nhánh này chuyên biệt để phát hiện và tái tạo các sự kiện bất thường (shocks).

- **Gating Network**: một mạng con (sub-net) quét phần *Residual* để tính xác suất xảy ra cú nhảy (\(\lambda_t\)).
- **Event Memory Bank**: bộ nhớ lưu trữ các vector learnable biểu diễn hình dáng của các loại cú sốc lịch sử (ví dụ giảm sốc, tăng vọt).
- **Mechanism**: Khi Gating kích hoạt, mô hình sẽ truy xuất (retrieve) mẫu jump phù hợp nhất từ bộ nhớ và cộng vào dự báo.

---

### 4. Adaptive Fusion Layer (Lớp Hợp Nhất)
Cân bằng giữa hai luồng tín hiệu bằng tham số học được \(\alpha\):

\[ Y_{final} = Y_{cont} + (1 + \tanh(\alpha)) \cdot Y_{jump} \]

- **Cross-Interaction**: sử dụng cơ chế attention nhẹ để luồng Continuous "nhận biết" được các cú Jump sắp tới và điều chỉnh cục bộ.

---

## Định nghĩa toán học
Mô hình xấp xỉ phương trình vi phân ngẫu nhiên (SDE) dạng Jump–Diffusion:

\[ dX_t = \underbrace{\mu(X_t, t)\,dt}_{\text{Continuous Stream (rKAN)}} + \underbrace{\sigma(X_t, t)\,dW_t}_{\text{Diffusion}} + \underbrace{J_t\,dN_t}_{\text{Jump Stream (Memory)}} \]

Trong đó:
- \(\mu(X_t, t)\): hàm trôi dạt (Drift), được học bởi rKAN.
- \(J_t\): kích thước cú nhảy (Jump size), lấy từ Event Memory Bank.
- \(N_t\): quá trình Poisson, được kiểm soát bởi Gating Network (xác suất xảy ra cú nhảy).

---