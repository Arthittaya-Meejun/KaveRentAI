# Model References

เอกสารนี้รวบรวมแหล่งอ้างอิงสำหรับอธิบายการออกแบบโมเดล Lease Probability
ในรายงาน โดยแยกเหตุผลของแต่ละส่วนออกจากกัน

## 1. Logistic Regression เป็น Baseline

Logistic Regression เหมาะเป็น Baseline สำหรับ Binary Classification เพราะให้
ค่าความน่าจะเป็นของคลาสบวก มีโครงสร้างไม่ซับซ้อน และใช้เปรียบเทียบกับโมเดล
non-linear ได้ง่าย การทดลองนี้ใช้ implementation แบบ regularized logistic
regression ของ scikit-learn

- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research, 12*, 2825–2830.
  https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf
- Scikit-learn developers. (n.d.). *LogisticRegression*.
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

## 2. CatBoost เป็นโมเดลเปรียบเทียบหลัก

CatBoost เป็น Gradient Boosting ที่รองรับความสัมพันธ์แบบ non-linear และ
categorical features งานต้นฉบับเสนอ ordered boosting และวิธีจัดการ categorical
features เพื่อลด prediction shift และ target leakage จาก target statistics
โค้ดในโปรเจกต์จึงส่งชื่อ categorical columns ผ่าน `cat_features` โดยตรง

- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A.
  (2018). CatBoost: Unbiased boosting with categorical features. In
  *Advances in Neural Information Processing Systems 31*.
  https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- CatBoost. (n.d.). *CatBoostClassifier*.
  https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier
- CatBoost. (n.d.). *Categorical features*.
  https://catboost.ai/docs/en/features/categorical-features

## 3. Fixed Four-week Prediction Horizon

ผลลัพธ์การปล่อยเช่าขึ้นกับเวลา จึงต้องกำหนดเวลาให้ชัดว่า “ปล่อยเช่าได้ภายใน
4 สัปดาห์หรือไม่” ประกาศช่วงท้ายที่ยังติดตามไม่ครบ 4 สัปดาห์ถูกตัดออก แทนการ
กำหนดให้เป็นคลาสลบ เพราะข้อมูลเหล่านั้นเป็น right-censored observations

แนวคิด time-dependent binary outcome และปัญหาการประเมินเมื่อมี censoring
อธิบายไว้โดย Heagerty, Lumley และ Pepe อย่างไรก็ตาม โปรเจกต์นี้ใช้วิธีง่ายคือ
วิเคราะห์เฉพาะแถวที่มี follow-up ครบ ไม่ได้สร้าง survival model

- Heagerty, P. J., Lumley, T., & Pepe, M. S. (2000). Time-dependent ROC
  curves for censored survival data and a diagnostic marker. *Biometrics,
  56*(2), 337–344. https://doi.org/10.1111/j.0006-341X.2000.00337.x

## 4. Chronological Split

ข้อมูลถูกแบ่ง Train, Validation และ Test ตามลำดับเวลา เพื่อไม่ให้โมเดลเรียนจาก
อนาคตแล้วกลับมาประเมินอดีต หลักการนี้สอดคล้องกับ time-series cross-validation
ซึ่งกำหนดให้ชุดฝึกอยู่ก่อนชุดประเมินเสมอ

- Scikit-learn developers. (n.d.). *TimeSeriesSplit*.
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

## 5. Evaluation Metrics

งานนี้เป็น probability prediction จึงใช้ Log Loss เป็น metric หลักในการเลือก
โมเดล และรายงาน ROC AUC กับ Average Precision สำหรับความสามารถในการจัดอันดับ
รวมทั้ง Balanced Accuracy และ F1 สำหรับผลหลังใช้ threshold

- Scikit-learn developers. (n.d.). *Metrics and scoring: Quantifying the
  quality of predictions*.
  https://scikit-learn.org/stable/modules/model_evaluation.html
- Scikit-learn developers. (n.d.). *log_loss*.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
- Scikit-learn developers. (n.d.). *Probability calibration*.
  https://scikit-learn.org/stable/modules/calibration.html

## ข้อความตัวอย่างสำหรับรายงาน

> งานวิจัยนี้ใช้ Logistic Regression เป็นโมเดลฐานสำหรับปัญหา Binary
> Classification และเปรียบเทียบกับ CatBoostClassifier ซึ่งสามารถเรียนรู้
> ความสัมพันธ์แบบไม่เป็นเส้นตรงและรองรับตัวแปรเชิงหมวดหมู่ได้
> (Prokhorenkova et al., 2018) ข้อมูลถูกแบ่งตามลำดับเวลาเพื่อป้องกันการใช้
> ข้อมูลอนาคตในการฝึกโมเดล และเลือกโมเดลจาก Validation Set ด้วย Log Loss
> เนื่องจากผลลัพธ์ที่ต้องการเป็นค่าความน่าจะเป็นของการปล่อยเช่าภายใน 4 สัปดาห์
