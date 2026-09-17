"""KaveRentAI Streamlit application entry point."""

import streamlit as st


st.set_page_config(page_title="KaveRentAI", page_icon="📈", layout="wide")

st.title("KaveRentAI")
st.subheader("AI & Decision Support System")
st.info(
    "ระบบรุ่นแรกอยู่ระหว่างการพัฒนา และใช้ข้อมูลจำลองเพื่อการศึกษาเท่านั้น"
)

left, right = st.columns(2)
with left:
    st.markdown("### Predictive Models")
    st.markdown(
        "- Rental Price Prediction\n"
        "- Time-to-Lease Prediction"
    )

with right:
    st.markdown("### Decision Support")
    st.markdown(
        "- Price Recommendation\n"
        "- Price-based Promotions\n"
        "- What-if Simulation\n"
        "- Estimated Revenue"
    )
