# Kịch bản thuyết trình
## Dự đoán tổn thương thận cấp (AKI) ở bệnh nhân DKA · Mô hình Random Forest

**Thời lượng gợi ý:** ~14–16 phút · **Đối tượng:** hội đồng kỹ thuật / y sinh
**Cách dùng:** mỗi mục dưới đây ứng với một slide. Phần in nghiêng ở cuối là câu *chuyển ý* sang slide sau. Số trong ngoặc là thời lượng gợi ý.

---

### Slide 1 — Trang bìa *(~45 giây)*

Xin chào quý vị. Hôm nay tôi xin trình bày một nghiên cứu thực nghiệm về **dự đoán tổn thương thận cấp — viết tắt là AKI — ở bệnh nhân nhiễm toan ceton do đái tháo đường, gọi tắt là DKA**, sử dụng mô hình Random Forest trên dữ liệu MIMIC-IV.

Bốn con số tóm tắt toàn bộ bài toán: chúng tôi làm việc với **970 mẫu huấn luyện**, mỗi bệnh nhân được mô tả bằng **77 đặc trưng** sau tiền xử lý, tỉ lệ ca dương tính là **38,9%**, và mô hình cuối đạt **AUC 0,807** trên tập kiểm thử nội bộ. Đây là bài toán phân loại nhị phân, nhãn `akdPositive` theo tiêu chuẩn KDIGO, trích xuất theo nghiên cứu của Fan và cộng sự năm 2023.

*Trước hết, hãy cùng làm rõ vì sao bài toán này lại có giá trị trên lâm sàng.*

---

### Slide 2 — Bối cảnh & bài toán *(~75 giây)*

Nhiễm toan ceton là một cấp cứu nội tiết – chuyển hoá rất thường gặp. Vấn đề là khi bệnh nhân DKA tiến triển thêm **tổn thương thận cấp**, nguy cơ tử vong, nguy cơ chuyển sang bệnh thận mạn và chi phí điều trị đều tăng lên đáng kể. Vì vậy, nếu ta dự đoán được sớm — ngay trong những giờ đầu khi bệnh nhân vào ICU — thì bác sĩ có thể phân tầng nguy cơ và can thiệp kịp thời.

Về mặt kỹ thuật, đây là **bài toán phân loại nhị phân**: đầu vào là thông tin bệnh nhân trong 24 giờ đầu ở ICU, đầu ra là dự đoán bệnh nhân có phát triển AKI trong vòng một tuần hay không.

Bài toán có ba thách thức chính. Thứ nhất là **mất cân bằng lớp** — chỉ 38,9% là ca dương tính, nên chỉ số accuracy rất dễ gây hiểu lầm; chúng tôi dùng ROC-AUC làm thước đo chính. Thứ hai là **dữ liệu chuỗi thời gian** có độ dài khác nhau giữa các bệnh nhân, phải tổng hợp về một biểu diễn cố định mà không đánh mất tín hiệu. Và thứ ba là **nguy cơ quá khớp**, vì tập dữ liệu khá nhỏ.

*Để hiểu rõ ba thách thức đó, ta cần nhìn vào chính tập dữ liệu.*

---

### Slide 3 — Tập dữ liệu (MIMIC-IV) *(~70 giây)*

Dữ liệu được trích xuất từ **MIMIC-IV** — một cơ sở dữ liệu chăm sóc tích cực công khai — theo đúng quy trình của Fan và cộng sự. Mỗi bản ghi tương ứng với một lần nằm ICU của một bệnh nhân DKA, và được lưu ở **định dạng JSON**.

Chúng ta có hai tệp. Tệp `train.json` gồm **970 bản ghi có nhãn**, trong đó 377 ca dương tính, tức 38,9%. Tệp `test.json` gồm **243 bản ghi nhưng không có nhãn**.

Điểm cần lưu ý ở đây — và nó định hình toàn bộ chiến lược về sau — là **tập test không có nhãn**, nên ta không thể dùng nó để đánh giá trực tiếp. Bắt buộc phải tách một phần dữ liệu huấn luyện ra làm tập kiểm thử nội bộ. Mỗi bệnh nhân được mô tả bằng **37 trường đo**, chia thành hai nhóm tĩnh và động.

*Hãy xem cụ thể 37 trường đo đó gồm những gì.*

---

### Slide 4 — Đặc trưng: tĩnh & động *(~70 giây)*

37 trường đo chia làm hai nhóm rõ rệt — và đây chính là **dữ liệu thô** đầu vào của khâu tiền xử lý.

Nhóm thứ nhất là **20 đặc trưng tĩnh**: mỗi trường chỉ có một giá trị duy nhất cho cả lần nằm viện — ví dụ tuổi, giới tính, chủng tộc, loại DKA, các điểm độ nặng như OASIS, SAPS-II, SOFA, độ lọc cầu thận eGFR, và các cờ bệnh nền.

Nhóm thứ hai là **17 đặc trưng động** — đây là các chuỗi thời gian được đo lặp lại trong 24 giờ đầu ICU, lưu dưới dạng cặp *thời điểm – giá trị*. Bao gồm các dấu hiệu sinh tồn như nhịp tim, nhịp thở, huyết áp, điểm Glasgow; và các xét nghiệm như BUN, creatinine, khoảng trống anion, đường huyết, bicarbonate, phosphate, canxi, tiểu cầu, hemoglobin, bạch cầu.

Chính sự tồn tại của nhóm động — với độ dài thay đổi — là lý do ta cần một bước tổng hợp thông minh, mà tôi sẽ nói kỹ ở phần điểm nhấn.

*Trước khi xây mô hình, ta phải quyết định chia dữ liệu thế nào cho đúng.*

---

### Slide 5 — Chiến lược phân chia dữ liệu *(~75 giây)*

Vì dữ liệu mất cân bằng, chúng tôi dùng **tách phân tầng** — stratified split — để giữ nguyên tỉ lệ dương tính khoảng 38,9% ở mọi tập con.

Cụ thể: từ 970 mẫu, chúng tôi tách ra **776 mẫu để huấn luyện** và **194 mẫu làm kiểm thử nội bộ** — tỉ lệ dương tính giữ ở 38,7%, gần như y hệt. Tập test 243 bản ghi dùng để dự đoán và nộp kết quả; lưu ý khi xuất kết quả, mỗi bệnh nhân chỉ giữ một bản ghi đầu tiên, nên 243 bản ghi rút còn **222 bệnh nhân**. Cuối cùng, sau khi đã chọn được cấu hình tốt nhất, chúng tôi **huấn luyện lại trên toàn bộ 970 mẫu** để tận dụng tối đa dữ liệu.

Có một nguyên tắc tôi muốn nhấn mạnh: **chống rò rỉ dữ liệu**. Bộ tiền xử lý chỉ được "học" — tức `fit` — trên tập train, rồi chỉ `transform` trên validation và test. Nhờ vậy thông tin từ tập đánh giá không bao giờ lọt ngược vào quá trình huấn luyện.

*Bây giờ là bức tranh tổng thể của toàn bộ hệ thống.*

---

### Slide 6 — Kiến trúc tổng thể *(~70 giây)*

Toàn bộ hệ thống gồm **hai mô-đun nối tiếp nhau**.

Bắt đầu từ `train.json`. Dữ liệu đi vào **Mô-đun A — tiền xử lý**, biến các bản ghi JSON thô thành một **ma trận số 970 × 77**, đồng thời lưu lại bộ biến đổi `preprocessor.joblib`. Ma trận này đi vào **Mô-đun B — Random Forest**: ta tách 80/20 phân tầng, dùng `class_weight = balanced`, rồi chạy **GridSearchCV** với 1.080 tổ hợp và 5-fold để chọn cấu hình theo ROC-AUC. Sau đó là bước **tinh chỉnh ngưỡng quyết định** theo ưu tiên lâm sàng, rồi **huấn luyện lại** cấu hình tốt nhất trên toàn bộ 970 mẫu.

Khi dự đoán, `test.json` đi qua đúng bộ tiền xử lý đó — chỉ `transform`, giữ nguyên thứ tự cột — để cho ra dự đoán xác suất và nhãn AKI, xuất ra file CSV kết quả.

*Ta sẽ đi sâu vào từng mô-đun, bắt đầu từ khâu tiền xử lý.*

---

### Slide 7 — Mô-đun A: Tiền xử lý dữ liệu *(~85 giây)*

Mô-đun A gồm năm bước, biến bản ghi JSON thô thành ma trận số chuẩn hoá.

**Bước một — làm phẳng.** Mỗi bệnh nhân thành một dòng. Đặc trưng tĩnh giữ nguyên; còn mỗi đặc trưng động được tổng hợp thành **ba thống kê: min, mean, max** — tức 17 biến động × 3 = 51 cột.

**Bước hai — điền 0 cho cờ bệnh nền vắng mặt.** Sáu cờ bệnh nền và giai đoạn bệnh thận mạn chỉ được ghi nhận khi bệnh nhân *có* bệnh. Vì vậy "vắng mặt" ở đây nghĩa là *không mắc* — đó là khuyết cấu trúc, nên điền 0 là đúng bản chất, không phải dữ liệu thiếu.

**Bước ba — loại biến khuyết trên 20% rồi KNN.** Trên dữ liệu này thực ra không biến nào vượt ngưỡng — tối đa chỉ khoảng 5,9% — nên ta giữ tất cả và điền giá trị thiếu bằng **KNNImputer với k = 5**.

**Bước bốn — mã hoá và chuẩn hoá.** Các biến phân loại như giới tính, bệnh gan, chủng tộc, loại DKA được one-hot hoặc mã hoá có thứ tự; các cột số được chuẩn hoá bằng StandardScaler.

**Bước năm — lưu và chống rò rỉ.** Toàn bộ imputer, scaler và danh sách cột được đóng gói vào `preprocessor.joblib`, và trên tập test ta chỉ `transform` chứ không bao giờ `fit` lại.

Kết quả là ma trận **970 × 77** sạch, cùng một bộ biến đổi tái sử dụng được, bảo đảm tái lập.

*Trong năm bước đó, bước một — tổng hợp min/mean/max — chính là đóng góp tôi muốn làm nổi bật.*

---

### Slide 8 — Điểm nhấn: biểu diễn động bằng min / mean / max *(~80 giây)*

Đây là điểm khác biệt so với bài báo gốc. Thay vì rút mỗi chuỗi thời gian về **một giá trị đơn**, chúng tôi đề xuất **giữ cả ba thống kê min, mean và max**.

Lý do nằm ở chỗ: các giá trị cực trị mang tín hiệu lâm sàng rất rõ. Đỉnh creatinine hay BUN, hoặc huyết áp thấp nhất, báo hiệu tình trạng giảm tưới máu thận. Nếu chỉ lấy trung bình, ta sẽ **làm phẳng và che giấu** chính những tín hiệu này.

Và đây là bằng chứng định lượng. Hãy nhìn tương quan giữa **huyết áp tâm thu** với nhãn AKI, tính riêng cho từng thống kê: giá trị **min cho tương quan −0,26**, **max cho +0,27** — hai tín hiệu mạnh và ngược chiều nhau. Nhưng nếu chỉ lấy **mean thì tương quan chỉ còn +0,05** — gần như bằng không. Nói cách khác, cùng một biến, giá trị trung bình đã **xoá gần hết** mối quan hệ mà min và max bộc lộ. Đó chính là lý do ta phải giữ cả ba.

*Có ma trận đặc trưng tốt rồi, giờ đến mô hình.*

---

### Slide 9 — Mô-đun B: Random Forest *(~75 giây)*

Mô-đun B là một mô hình **học tập hợp** — gồm nhiều cây quyết định — nhận đầu vào ma trận 970 × 77 và trả về xác suất AKI cho mỗi bệnh nhân.

Random Forest hoạt động dựa trên bốn cơ chế. **Bagging**: mỗi cây được huấn luyện trên một mẫu bootstrap khác nhau, tạo sự đa dạng. **Feature subsampling**: tại mỗi nút chỉ xét một tập con đặc trưng ngẫu nhiên — ở đây khoảng log₂77 ≈ 6 đặc trưng — để khử tương quan giữa các cây. Tiêu chí chia là **độ vẩn Gini**, kết hợp `class_weight='balanced'` để phạt nặng lỗi ở lớp thiểu số. Cuối cùng, dự đoán là **trung bình xác suất** trên toàn bộ cây, giúp giảm phương sai so với một cây đơn lẻ.

Vì sao chọn Random Forest? Vì nó **ổn định ngay khi chưa tinh chỉnh**, ít nhạy với siêu tham số, bắt được quan hệ phi tuyến và tương tác đặc trưng, lại **diễn giải được** qua feature importance — và là một mô hình nền mạnh để so sánh công bằng.

*"Ít nhạy với siêu tham số" — vậy việc tinh chỉnh có thực sự đáng không? Ta đã khảo sát rất kỹ.*

---

### Slide 10 — Tinh chỉnh siêu tham số (GridSearchCV) *(~75 giây)*

Chúng tôi khảo sát **toàn diện** không gian siêu tham số, và quan trọng là **ghi lại mọi cấu hình đã thử** để có thể kiểm chứng.

Không gian tìm kiếm có sáu chiều: số cây `n_estimators`, độ sâu tối đa `max_depth`, hai tham số kiểm soát phân nhánh, `max_features`, và bật/tắt bootstrap. Nhân các lựa chọn lại được **1.080 tổ hợp**; mỗi tổ hợp chạy kiểm định chéo 5-fold, thành **5.400 lần khớp mô hình**, mất khoảng **45 phút trên 6 nhân CPU**. Tiêu chí chọn là ROC-AUC.

Về tính tái lập: chúng tôi dùng StratifiedKFold để giữ tỉ lệ lớp trong mỗi fold, cố định `class_weight='balanced'` để cô lập ảnh hưởng của các tham số cấu trúc, đặt **`random_state = 42`** để tái lập hoàn toàn, và lưu toàn bộ bảng kết quả vào `rf_grid_search_all.csv`.

*Và đây là kết quả — có lẽ khiêm tốn hơn nhiều người kỳ vọng.*

---

### Slide 11 — Kết quả tinh chỉnh *(~80 giây)*

Sau 1.080 tổ hợp, chỉ có **đúng hai "núm vặn" thay đổi** so với cấu hình mặc định.

Cấu hình baseline — 300 cây, `max_features='sqrt'` — cho CV ROC-AUC **0,8042**. Cấu hình tốt nhất — 500 cây, `max_features='log2'` — cho **0,8109**, tức nhỉnh hơn **0,0067**. Đáng chú ý là `max_depth` vẫn được giữ ở `None`.

Và tôi muốn đọc kết quả này một cách **trung thực**. Việc giữ `max_depth=None` cho thấy: trên tập dữ liệu nhỏ này, chính cơ chế trung bình hoá của rừng đã kiểm soát phương sai, nên không cần cắt sâu cây. Phần cải thiện đến từ việc tăng nhẹ số cây và lấy mẫu đặc trưng hẹp hơn — log₂77 ≈ 6 so với √77 ≈ 9 — giúp khử tương quan thêm.

Điều quan trọng nhất: **độ lệch chuẩn của CV, khoảng 0,044, còn lớn hơn cả mức cải thiện 0,0067**. Nên cách đọc đúng không phải là "tinh chỉnh giúp tăng vọt", mà là: cấu hình tinh chỉnh **ít nhất tốt ngang** baseline và **nhỉnh hơn về trung bình**. Đây đúng với bản chất "mạnh sẵn" của Random Forest.

*Nếu siêu tham số chỉ nhích nhẹ, thì đâu mới là đòn bẩy thật sự cho lâm sàng? Câu trả lời là ngưỡng quyết định.*

---

### Slide 12 — Tinh chỉnh ngưỡng quyết định *(~85 giây)*

Trên lâm sàng, **bỏ sót một ca AKI — tức âm tính giả — nguy hại hơn nhiều so với một báo động giả**. Vì vậy ta ưu tiên **độ nhạy**. Và độ nhạy được điều khiển bằng *ngưỡng quyết định*, chứ không phải bằng siêu tham số.

Trên 194 bệnh nhân kiểm thử nội bộ, chúng tôi quét ngưỡng và thấy: ở **ngưỡng mặc định 0,50**, độ nhạy là 0,667. Chuyển sang **ngưỡng 0,47** — điểm tối ưu F1 — độ nhạy lên **0,707** và F1 lên 0,697 mà **gần như không mất độ đặc hiệu**. Còn nếu thực sự ưu tiên lâm sàng, **ngưỡng 0,33** đẩy độ nhạy lên **0,853** — tức bắt được khoảng 85% số ca AKI — đổi lại độ đặc hiệu giảm còn 0,555.

Kết luận then chốt của slide này: **ngưỡng là đòn bẩy cải thiện độ nhạy mạnh hơn cả việc tinh chỉnh siêu tham số.** Cùng một mô hình, chỉ cần dịch điểm vận hành là ta đã thay đổi đáng kể hành vi lâm sàng của nó.

*Giờ hãy đặt kết quả của chúng tôi cạnh các nghiên cứu liên quan.*

---

### Slide 13 — Kết quả & đánh giá *(~80 giây)*

So với các kết quả liên quan, mô hình của chúng tôi rất cạnh tranh. Random Forest baseline đạt AUC **0,806**, bản tinh chỉnh **0,807** — **tương đương** mô hình XGBoost tốt nhất của Fan và cộng sự năm 2023, vốn ở mức khoảng 0,800 — nhưng đạt được bằng một **cách biểu diễn đặc trưng và cách chia dữ liệu khác**.

Về **độ quan trọng đặc trưng**, nhóm đóng góp lớn nhất là các điểm độ nặng SAPS-II, OASIS, SOFA, cùng với eGFR, tuổi, và các thống kê của BUN và cân nặng — hoàn toàn **phù hợp** với phân tích tương quan EDA và với bài báo gốc, cho thấy mô hình học đúng tín hiệu lâm sàng.

Nhưng tôi cũng phải nói thẳng về **mặt hạn chế**: mô hình **vẫn còn quá khớp**. AUC trên tập train xấp xỉ 1,0 — cây sâu gần như học thuộc dữ liệu huấn luyện — trong khi validation là 0,807. Việc tinh chỉnh có cải thiện khả năng tổng quát nhưng **chưa thu hẹp được khoảng cách này**. Đây chính là điểm cần xử lý ở phiên bản sau.

*Và đó là cầu nối tự nhiên sang phần kết luận cùng định hướng.*

---

### Slide 14 — Kết luận & định hướng *(~75 giây)*

Tóm lại, nghiên cứu này đóng góp một **quy trình thực nghiệm chặt chẽ, minh bạch và tái lập được**, với ba điểm chính.

Một, **tiền xử lý chống rò rỉ** — từ flatten min/mean/max, KNN, mã hoá và chuẩn hoá, đến đóng gói `preprocessor.joblib`. Hai, **tinh chỉnh có hệ thống và kiểm chứng được** — 1.080 tổ hợp GridSearchCV, CV ROC-AUC từ 0,8042 lên 0,8109, AUC test 0,807. Ba — và là bài học thực tiễn nhất — **ngưỡng quyết định mới là đòn bẩy lâm sàng mạnh nhất**, đẩy độ nhạy lên tới 0,707 hoặc 0,853 tuỳ điểm vận hành.

Về **định hướng tương lai**, chúng tôi sẽ: dùng **LASSO** để chọn đặc trưng và giảm chiều; **so sánh XGBoost** tinh chỉnh trên cùng các fold và độ đo; **hiệu chỉnh xác suất** bằng CalibratedClassifierCV cho quyết định lâm sàng; áp dụng **kiểm định chéo lồng** để ước lượng không thiên lệch; và xem lại **chính sách gộp trùng** theo hướng chọn lần ICU nguy cơ cao nhất.

Xin cảm ơn quý vị đã lắng nghe. Tôi rất sẵn lòng trả lời các câu hỏi.

---

*Hết kịch bản — chúc bạn thuyết trình thành công.*
