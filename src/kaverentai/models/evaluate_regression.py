"""ฟังก์ชันสำหรับประเมินโมเดล Regression."""

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression(y_true, y_pred, model_name="Model"):
    """คำนวณและแสดงผล MAE, RMSE และ R2."""

    # 1. คำนวณค่า MAE
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    # 2. คำนวณค่า RMSE
    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    # 3. คำนวณค่า R2
    r2 = r2_score(
        y_true,
        y_pred,
    )

    # 4. รวมผลลัพธ์ไว้ใน Dictionary
    results = {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    # 5. แสดงผลลัพธ์
    print(f"\n{model_name}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))

    # 6. ส่งผลลัพธ์กลับไปให้ไฟล์ Train ใช้ต่อ
    return results
