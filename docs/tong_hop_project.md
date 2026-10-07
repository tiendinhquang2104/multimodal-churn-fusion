Đến **06/10/2026**, Project Master Thesis đã chốt khá rõ **bài toán, nguyên tắc dữ liệu và cách đánh giá**. **Novelty cụ thể và kiến trúc cuối cùng vẫn đang được lựa chọn.** Dưới đây là bản tổng hợp, phân biệt quyết định của bạn với phương án mới được đề xuất.

**1. Những nội dung đã chốt**

| Nội dung | Quyết định / yêu cầu đã thống nhất |
|---|---|
| **Hướng đề tài** | **Multimodal Customer Churn Prediction**, gắn với **hệ thống hỗ trợ ra quyết định — DSS**. |
| **Trọng tâm nghiên cứu** | Thiết kế phương pháp **fusion** để kết hợp dữ liệu giao dịch và các phương thức bổ sung, hướng tới dự đoán churn và giải thích kết quả. |
| **Cách chứng minh** | Chứng minh bằng **thực nghiệm, so sánh baseline và ablation**; không đặt trọng tâm vào phát triển lý thuyết toán học phức tạp. |
| **Yêu cầu về đóng góp** | **Concat hoặc generic gate riêng lẻ không đủ để gọi là novelty.** Cần kiểm tra nghiên cứu liên quan, đặc biệt giai đoạn 2025–2026, trước khi khẳng định research gap. |
| **Ưu tiên triển khai** | Dễ triển khai, tiết kiệm thời gian và tài nguyên, nhưng vẫn có đóng góp nghiên cứu đủ rõ. |
| **Phương thức dữ liệu mục tiêu** | **Transaction/tabular + text + image**; **audio không bắt buộc**. |
| **Tổ chức thực nghiệm** | Ưu tiên **một dataset chính** để chạy đầy đủ ablation, robustness, resource và explainability. Dataset khác chỉ dùng kiểm chứng đại diện, không lặp toàn bộ thực nghiệm. |
| **Tên project/repository** | **`multimodal-churn-fusion`**. |

**2. Định nghĩa churn và nguyên tắc dữ liệu**

Đây là phần đã xác lập rõ và cần giữ xuyên suốt:

- **Churn phải dựa trên giao dịch hoặc thuê bao thực tế.** Không dùng việc ngừng review, ngừng click hoặc ngừng hoạt động trên nền tảng đánh giá làm nhãn churn.
- Với **contractual churn**: không gia hạn hoặc chấm dứt thuê bao sau thời điểm hết hạn và khoảng gia hạn được định nghĩa.
- Với **non-contractual churn**: không có giao dịch hợp lệ trong **cửa sổ dự đoán tương lai**, tính sau một thời điểm cắt dữ liệu.
- **Input chỉ lấy từ observation window; label được tạo từ future prediction window**, để tránh rò rỉ thông tin tương lai.
- Dữ liệu cần liên kết được ở **cấp khách hàng**, có mốc thời gian và điều kiện sử dụng rõ ràng.
- Mục tiêu ban đầu là tìm **tối đa ba dataset phù hợp**, nhưng không ép đủ ba nếu không đáp ứng các tiêu chí bắt buộc.

**Chưa chốt độ dài cụ thể của các cửa sổ thời gian.** Những giá trị như churn 90 ngày vẫn cần được lựa chọn và kiểm chứng trên dữ liệu.

**3. Cách đánh giá giá trị của từng modality**

Bạn đã xác định các cấu hình ablation quan trọng:

| Cấu hình | Mục đích |
|---|---|
| **Transaction only** | Đo năng lực dự đoán từ lịch sử giao dịch. |
| **Transaction + Text** | Kiểm tra giá trị bổ sung của văn bản. |
| **Transaction + Image** | Kiểm tra giá trị bổ sung của hình ảnh. |
| **Transaction + Text + Image** | Kiểm tra hiệu quả khi kết hợp đầy đủ. |

**Text-only và Image-only không bắt buộc.** Giao dịch là phần cốt lõi của bài toán; thực nghiệm cần cho thấy text và image bổ sung được gì.

Kế hoạch có nhắc đến **E0–E8**, nhưng chưa đủ căn cứ để ghi lại chính xác từng mã thí nghiệm. Vì vậy, chưa nên xem bảng ánh xạ E0–E8 là đã chốt.

**4. Định hướng baseline mới nhất**

Trong trao đổi ngày **06/10**, bạn ưu tiên baseline theo hướng **Transformer/self-attention**. Phương án đang được đề xuất là:

| Thành phần | Định hướng hiện tại |
|---|---|
| Transaction encoder | **Transaction Transformer** |
| Text encoder | **MiniLM/BERT** |
| Image encoder | **CLIP-ViT** |
| Biểu diễn trước fusion | Các nhánh tạo biểu diễn riêng, rồi chiếu về **128D** |
| Fusion baseline | **Concatenation** |
| Prediction head | **MLP → churn risk** |

Đây là **baseline để so sánh**, chưa phải đóng góp mới của luận văn. Lựa chọn chính xác encoder và kích thước 128D vẫn thuộc phương án triển khai.

Kiến trúc trước đó với **cross-attention → reliability gate → weighted fusion → SHAP/modality weight** đã được thảo luận, nhưng **chưa có căn cứ coi đó là kiến trúc cuối cùng đã chốt**.

**5. Dataset: đã khảo sát, chưa xác nhận lựa chọn cuối cùng**

Báo cáo đính kèm đã khảo sát **KKBox, H&M, Complete Journey và Olist**. Các thứ hạng trong báo cáo là kết quả đánh giá tham khảo, không phải quyết định chọn dataset.

| Dataset | Vai trò đã được đề xuất | Trạng thái |
|---|---|---|
| **H&M** | Dataset chính cho transaction + text + image | Được ưu tiên trong các đề xuất gần đây; chưa thấy xác nhận cuối cùng của bạn. |
| **KKBox** | Kiểm chứng trên bài toán churn thuê bao | Đề xuất; cần phân biệt dữ liệu nhiều nguồn với text/image/audio thực sự. |
| **Complete Journey** | Kiểm chứng trên lịch sử mua hàng | Đề xuất; chưa chốt cách sử dụng trong benchmark multimodal. |
| **Olist** | Thử nghiệm bổ sung với dữ liệu mua lặp thưa | Optional, chưa chốt. |

Các tỷ lệ và thống kê trong báo cáo cần được kiểm tra lại từ dữ liệu hoặc nguồn gốc trước khi đưa vào luận văn.

**6. Novelty: đang nghiêng về hướng nào?**

Bạn đang **nghiêng về Semantic Preference Drift**, đồng thời yêu cầu hướng nghiên cứu phải dễ triển khai và có gap đủ tốt.

Ý tưởng đã được đề xuất là: dùng text và image để biểu diễn ý nghĩa sản phẩm đã mua, so sánh **sở thích lịch sử với sở thích gần đây**, rồi kiểm tra xem sự thay đổi đó có giúp dự đoán churn hay không.

Tuy nhiên:

- **Semantic Preference Drift chưa được chốt là contribution cuối cùng.**
- **RAMF-Churn** và **Transaction-Centric Hierarchical Multimodal Fusion** là các hướng/tên đã thảo luận.
- **Event-aligned fusion** được đề xuất làm nền tảng tổ chức dữ liệu hoặc baseline.
- Chưa hoàn tất kiểm tra prior art để tuyên bố novelty.

**7. Tài liệu và trạng thái triển khai**

Bài **Deep Multimodal Data Fusion — Zhao et al. (2024)** là tài liệu tham khảo chính đã được đưa vào project. Bài phân loại thành **năm nhóm**: Encoder–Decoder, Attention Mechanism, Graph Neural Network, Generative Neural Network và Constraint-based methods. Danh sách bảy nhóm từng thảo luận là khung tổng hợp rộng hơn, không phải nguyên văn taxonomy của bài này. 2024-FEI ZHAO-Deep Multimodal D…

Bạn đã chọn tên repo và yêu cầu **prompt khởi tạo project**. Chưa có bằng chứng xác nhận repo đã được tạo, pipeline đã chạy hoặc mô hình đã train. **Khoảng 500 Colab CU** mới là mức ngân sách được hỏi và thảo luận, chưa phải khoản đã xác nhận mua.

Các điểm còn cần chốt là **dataset chính, cửa sổ tạo nhãn, temporal split, contribution cuối cùng, cấu hình thực nghiệm và tên đề tài chính thức**.