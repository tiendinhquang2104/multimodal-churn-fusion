Lý do tôi chọn ba kiến trúc đó là vì **ba modality không bắt đầu từ cùng một mức thông tin**. Text và image đã có pretrained encoder rất mạnh; transaction thì không. Vì vậy không nên cho cả ba nhánh cùng một kiến trúc sâu như nhau.

Điểm quan trọng trước tiên: các cấu hình dưới đây là **starting architecture hợp lý để làm thesis và ablation**, không phải những con số “tối ưu về mặt lý thuyết”.

## 1. Text: MiniLM-L6-v2 → Projection 128 → 2-layer Temporal Transformer → Attention Pool → 128D

Ta phải tách hai nhiệm vụ khác nhau:

```text
Nhiệm vụ A:
Hiểu nội dung của từng sản phẩm

Nhiệm vụ B:
Hiểu lịch sử các sản phẩm customer đã mua
```

MiniLM xử lý nhiệm vụ A, Temporal Transformer xử lý nhiệm vụ B.

Ví dụ customer mua:

```text
t1: black dress
t2: black jacket
t3: formal trousers
t4: sportswear
```

Đầu tiên:

```text
"black dress"
      ↓
   MiniLM
      ↓
text embedding của sản phẩm 1
```

MiniLM đã được pretrained trên text nên chúng ta **không cần một Transformer lớn tự học ngôn ngữ từ đầu**.

### Tại sao MiniLM-L6-v2?

Vì tôi muốn một encoder:

- đã pretrained;
- tương đối nhẹ;
- nhanh khi phải encode hàng chục/hàng trăm nghìn article;
- sinh sentence embedding trực tiếp;
- dễ freeze để giảm computational cost.

Nếu dùng BERT-base cũng được, nhưng với mục tiêu hiện tại MiniLM thực dụng hơn.

Bạn chưa cần chứng minh:

> MiniLM là model text tốt nhất.

Bạn chỉ cần một **strong pretrained backbone ổn định** để tập trung nghiên cứu vào fusion.

---

### Tại sao cần `Projection → 128D`?

MiniLM không tự nhiên trả đúng 128 chiều.

Giả sử output là:

\[
e_{text}\in R^{384}
\]

ta dùng:

\[
Linear(384\rightarrow128)
\]

để có:

\[
e'_{text}\in R^{128}
\]

Lý do thứ nhất là đồng nhất cả ba modality:

```text
Text        → 128D
Image       → 128D
Transaction → 128D
```

Điều này rất tiện cho fusion về sau.

Lý do thứ hai là giảm kích thước feature table.

Nếu để:

```text
text = 384D
image = 512D
transaction = 128D
```

thì image/text có số chiều lớn hơn transaction rất nhiều.

Không có nghĩa chúng chắc chắn “áp đảo” mathematically, nhưng representation space bị mất cân đối và việc xây các module fusion phía sau phức tạp hơn.

128D là một mức khá hợp lý giữa:

```text
quá nhỏ                 hợp lý                   quá lớn
32/64D                   128D                  512/768D
```

Tôi sẽ coi `128` là hyperparameter để sau này ablation:

```text
64
128
256
```

chứ không claim 128 là tối ưu.

---

### Tại sao thêm Temporal Transformer sau MiniLM?

Vì MiniLM chỉ biết:

> sản phẩm này nói về cái gì?

Nó không biết:

> customer đã mua các sản phẩm này theo thứ tự nào?

Ví dụ:

```text
Customer A:
formal → formal → formal → casual

Customer B:
casual → formal → formal → formal
```

Nếu average:

\[
\frac{e_1+e_2+e_3+e_4}{4}
\]

hai history có thể rất giống nhau.

Nhưng sequence khác nhau.

Nên:

```text
MiniLM(article 1) ─┐
MiniLM(article 2) ─┤
MiniLM(article 3) ─┤ → Temporal Transformer
MiniLM(article 4) ─┘
```

Temporal Transformer học quan hệ:

\[
e_1 \leftrightarrow e_2 \leftrightarrow e_3 \leftrightarrow e_4
\]

trong lịch sử customer.

---

### Tại sao chỉ 2 layers?

Vì MiniLM đã làm phần semantic nặng nhất rồi.

Temporal Transformer chỉ cần học:

```text
"Customer mua sản phẩm nào trước/sau sản phẩm nào?"
```

chứ không cần học lại tiếng Anh.

Nếu dùng:

```text
6–12 temporal layers
```

tôi thấy quá nặng cho bước này, tăng:

- compute;
- memory;
- nguy cơ overfit;
- thời gian tuning.

Nên tôi bắt đầu bằng:

```text
2 layers
```

rồi ablation:

```text
1 layer
2 layers
3 layers
```

Nếu 3 layers không tốt hơn thì rất dễ bảo vệ việc giữ 2.

---

## 2. Image: CLIP ViT-B/32 → Projection 128 → 2-layer Temporal Transformer → Attention Pool → 128D

Logic gần tương tự text.

CLIP xử lý:

> ảnh của sản phẩm này có visual semantics gì?

Temporal Transformer xử lý:

> visual preference của customer thay đổi như thế nào qua lịch sử?

Ví dụ:

```text
t1 → dark clothing
t2 → dark clothing
t3 → formal clothing
t4 → colourful casual clothing
```

Một ảnh đơn lẻ:

```text
image
 ↓
CLIP ViT
 ↓
visual embedding
```

CLIP không biết lịch sử customer.

Do đó:

```text
Image 1 → CLIP → v1
Image 2 → CLIP → v2
Image 3 → CLIP → v3
Image 4 → CLIP → v4
                     ↓
             Temporal Transformer
```

---

### Tại sao CLIP ViT-B/32?

Không phải vì đây là vision model mạnh nhất hiện nay.

Tôi chọn nó vì:

- pretrained rất phổ biến;
- embedding semantic tốt;
- đã được học alignment image-text;
- tương đối nhẹ;
- dễ sử dụng như frozen feature extractor;
- phù hợp để làm baseline/research backbone.

Trong H&M, đây là điểm đặc biệt hữu ích vì text và image của cùng một product có thể nằm trong những semantic space tương đối tương thích hơn so với việc chọn hai encoder hoàn toàn độc lập.

Ví dụ:

```text
Image: black leather jacket
Text:  "black biker jacket"
```

CLIP đã được huấn luyện với tư tưởng image ↔ text alignment.

Điều đó rất có lợi nếu sau này bạn nghiên cứu multimodal fusion.

---

### Tại sao không dùng raw ViT?

Bạn có thể.

Nhưng:

```text
ViT pretrained ImageNet
```

chủ yếu học visual classification features.

Trong khi:

```text
CLIP
```

được học từ image-text pairs.

Vì thesis của bạn có cả:

```text
image + text
```

CLIP hợp với câu chuyện multimodal hơn.

---

### Tại sao cũng 2 temporal layers?

Cùng lý do text.

CLIP đã xử lý:

```text
raw pixels → semantic representation
```

Temporal Transformer chỉ xử lý:

```text
semantic representation 1
semantic representation 2
...
semantic representation N
            ↓
purchase history dynamics
```

Nó không phải học thị giác từ đầu.

Do đó tôi không thấy cần quá sâu ngay từ đầu.

---

# 3. Transaction: Event embedding → 3-layer Transformer → Attention Pool → 128D

Transaction khác hoàn toàn.

Với text:

```text
MiniLM = pretrained knowledge
```

Với image:

```text
CLIP = pretrained knowledge
```

Còn transaction:

```text
price
channel
time gap
category
...
```

không có một pretrained Transformer phổ quát tương đương MiniLM/CLIP mà bạn chỉ việc lấy về dùng.

Vì vậy branch này phải làm nhiều việc hơn.

---

## Bước đầu tiên: Event Embedding

Giả sử transaction:

```text
2020-05-01
price = 0.035
channel = 2
article = jacket
```

Không thể đưa trực tiếp string/numbers lung tung vào Transformer.

Ta phải biến thành vector:

```text
price ────────────────┐
delta_time ───────────┤
channel embedding ────┤
category embedding ───┤
                      ↓
              event representation
```

Ví dụ:

\[
x_t\in R^{128}
\]

Mỗi transaction lúc này trở thành một token tương tự token trong NLP.

Trong NLP:

```text
"The"
"customer"
"left"
```

→ token embeddings.

Ở đây:

```text
transaction 1
transaction 2
transaction 3
```

→ event embeddings.

Transformer sau đó đọc:

\[
[x_1,x_2,\ldots,x_N]
\]

---

## Tại sao Transaction dùng 3 layers thay vì 2?

Vì transaction branch không có pretrained semantic encoder mạnh ở phía trước.

Text:

```text
MiniLM
 ↓
đã hiểu semantic
 ↓
2 temporal layers
```

Image:

```text
CLIP
 ↓
đã hiểu visual semantic
 ↓
2 temporal layers
```

Transaction:

```text
raw event features
 ↓
event embedding
 ↓
Transformer phải tự học behavioral pattern
```

Nó phải học nhiều quan hệ hơn:

```text
frequency
recency
price behavior
time gaps
channel changes
product/category transition
long-term pattern
short-term pattern
```

Vì vậy tôi cho nó một chút capacity lớn hơn:

```text
3 layers
```

chứ không phải vì có quy luật:

> transaction Transformer bắt buộc phải có 3 layer.

Thực nghiệm vẫn nên kiểm tra:

```text
2
3
4
```

Tôi dự đoán 3 là một starting point hợp lý.

---

# 4. Tại sao `d_model = 128`?

Đây là một quyết định khá quan trọng.

Một transaction token sẽ là:

\[
x_t\in R^{128}
\]

Transformer giữ:

\[
d_{model}=128
\]

và output customer cũng:

\[
z_{transaction}\in R^{128}
\]

Như vậy toàn pipeline rất sạch:

```text
event → 128
Transformer → 128
pool → 128
```

Không cần:

```text
128 → 256 → 512 → 128
```

vô ích ở baseline.

Và cuối cùng:

```text
Transaction 128D
Text        128D
Image       128D
```

rất thuận tiện cho feature table.

---

# 5. Tại sao 4 attention heads?

Multi-head attention chia 128 chiều thành các head.

Nếu:

\[
d_{model}=128
\]

và:

\[
n_{heads}=4
\]

thì mỗi head có:

\[
128/4=32
\]

chiều.

Đây là cấu hình khá cân bằng.

Có thể hình dung mỗi head **có khả năng** học những interaction khác nhau, ví dụ:

```text
Head 1 → short-term relation
Head 2 → category transition
Head 3 → price/purchase pattern
Head 4 → long-distance relation
```

Không được hiểu cứng rằng từng head chắc chắn học đúng những thứ này; đó chỉ là trực giác.

Tại sao không 8 heads?

Với:

\[
128/8=16
\]

mỗi head khá nhỏ.

Và model phức tạp hơn trong khi chưa chắc mang lại lợi ích.

Tại sao không 1 head?

Bạn mất khả năng nhìn sequence bằng nhiều attention subspaces.

Nên:

```text
4 heads
```

là điểm bắt đầu hợp lý.

Sau này:

```text
2 heads
4 heads
8 heads
```

là một ablation đơn giản.

---

# 6. Tại sao Attention Pooling?

Đây là phần rất quan trọng.

Transformer sẽ trả về:

\[
h_1,h_2,\ldots,h_N
\]

tức mỗi transaction/product vẫn có một output vector.

Nhưng feature table cần:

\[
1 customer \rightarrow 1 vector
\]

nên phải pool.

---

### Mean pooling

Cách đơn giản:

\[
z=\frac{1}{N}\sum h_i
\]

Nghĩa là mọi event quan trọng bằng nhau.

Nhưng với churn/customer behavior:

```text
transaction cách đây 1 năm
```

có thể không quan trọng bằng:

```text
3 transaction gần nhất
```

hoặc có những purchase rất đặc trưng.

---

### Attention pooling

Cho model học:

\[
\alpha_1,\alpha_2,...,\alpha_N
\]

với:

\[
\sum_i\alpha_i=1
\]

rồi:

\[
z=\sum_i \alpha_i h_i
\]

Ví dụ:

```text
T1  → weight 0.03
T2  → weight 0.05
T3  → weight 0.07
T4  → weight 0.20
T5  → weight 0.65
```

Model có thể đánh trọng số cao hơn cho những event hữu ích.

Đây là lý do tôi thích attention pooling hơn mean pooling.

---

# 7. Tại sao không chỉ dùng `[CLS]`?

Cũng hoàn toàn có thể.

Ta thêm một token:

```text
[CLS], T1, T2, T3, T4
```

và dùng output:

```text
h_CLS
```

làm customer vector.

Đây là một baseline tốt.

Tôi chọn attention pooling vì nó trực quan hơn cho thesis:

> model học event nào đóng góp nhiều vào customer representation.

Bạn thậm chí có thể visualize:

```text
transaction 1 → 5%
transaction 2 → 12%
transaction 3 → 60%
...
```

Nó giúp phần interpretability.

---

# 8. Toàn bộ ba branch khác nhau ở đâu?

Điểm cốt lõi là:

| Branch | Ai học semantic ban đầu? | Ai học history? |
|---|---|---|
| Text | MiniLM | Temporal Transformer |
| Image | CLIP | Temporal Transformer |
| Transaction | chính Event Encoder + Transformer | Transformer |

Nên:

```text
TEXT
raw text
 ↓
PRETRAINED MiniLM
 ↓
semantic product vectors
 ↓
small temporal Transformer
 ↓
customer text vector
```

```text
IMAGE
raw image
 ↓
PRETRAINED CLIP
 ↓
semantic product vectors
 ↓
small temporal Transformer
 ↓
customer image vector
```

```text
TRANSACTION
raw structured event
 ↓
event encoding
 ↓
Transformer tự học behavioral representation
 ↓
customer transaction vector
```

Đó là lý do transaction branch sâu hơn một chút.

---

# 9. Vì sao cuối cùng cả ba đều 128D?

Vì tôi muốn bảng feature của bạn có tính đối xứng:

\[
z_i^{T}\in R^{128}
\]

\[
z_i^{X}\in R^{128}
\]

\[
z_i^{I}\in R^{128}
\]

rồi:

\[
z_i=
[z_i^T;z_i^X;z_i^I]
\in R^{384}
\]

Feature table:

```text
customer_id
transaction_000 ... transaction_127
text_000        ... text_127
image_000       ... image_127
```

Sau này bạn có thể làm baseline:

```text
384D
 ↓
XGBoost / MLP / classifier
 ↓
churn
```

và proposed fusion:

```text
128D   128D   128D
  ↓      ↓      ↓
 multimodal fusion
       ↓
     churn
```

Rất sạch về thiết kế thực nghiệm.

---

## Nhưng có một điều tôi sẽ thay đổi trong cách gọi

Tôi sẽ không gọi:

> `MiniLM → Projection → Temporal Transformer`

là toàn bộ một **Text Transformer**.

Chính xác hơn nên gọi:

> **Text semantic encoder + customer-history temporal encoder**

Tương tự image:

> **Visual semantic encoder + customer-history temporal encoder**

Vì CLIP và MiniLM encode từng article, còn Transformer phía sau encode **customer history**.

Đây là distinction rất quan trọng khi bạn giải thích với giảng viên.

---

### Kiến trúc tôi sẽ chốt làm Version 1

```text
TEXT
Article text
→ frozen MiniLM-L6-v2
→ Linear(384,128)
→ positional/time encoding
→ TransformerEncoder ×2
→ attention pooling
→ 128D
```

```text
IMAGE
Article image
→ frozen CLIP ViT-B/32
→ Linear(512,128)
→ positional/time encoding
→ TransformerEncoder ×2
→ attention pooling
→ 128D
```

```text
TRANSACTION
price + channel + Δtime + categorical features
→ event embedding 128D
→ positional/time encoding
→ TransformerEncoder ×3
   d_model=128
   heads=4
   FFN=256/512
→ attention pooling
→ 128D
```

Và tôi sẽ giữ các tham số `128D`, `2/3 layers`, `4 heads` như **baseline hyperparameters cần kiểm chứng bằng ablation**, chứ không viết trong thesis rằng chúng là lựa chọn tối ưu tuyệt đối.

Điểm quan trọng nhất trong toàn bộ lựa chọn này là: **MiniLM và CLIP chịu trách nhiệm hiểu nội dung; Transformer phía sau chịu trách nhiệm hiểu lịch sử customer; transaction Transformer phải tự học hành vi nên cần capacity lớn hơn một chút.**