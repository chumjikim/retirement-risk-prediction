import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

st.set_page_config(page_title="퇴직연금 리스크 예측", layout="wide")
st.title("퇴직연금 재정검증 리스크 예측 결과 (2024)")

# ===============================
# 1) 모델 & 데이터 로드
# ===============================
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("final_model.json")
    with open("final_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta

@st.cache_data
def load_data():
    df = pd.read_csv("prediction_2024_extended.csv")
    df.columns = df.columns.str.strip()  # 👈 공백 제거
    return df

model, meta = load_model()
df = load_data()

# ===============================
# 2) 검색 기능
# ===============================
company = st.text_input("🔍 사업자번호 또는 업체명 검색")

if company:
    result = df[
        df["사업자번호"].astype(str).str.contains(company) |
        df["업체명"].astype(str).str.contains(company)
    ]
    if result.empty:
        st.warning("검색 결과 없음")
else:
    result = df.copy()

# ===============================
# 3) 최종 판정 (70% 기준)
# ===============================
result["final_judgement"] = result.apply(
    lambda r: 0 if r["p_risk"] >= 0.7 else 1, axis=1
)

# 📊 전체 통계
st.markdown("### 전체 통계")
col1, col2, col3 = st.columns(3)
col1.metric("전체 기업 수", len(result))
col2.metric("리스크 기업 수", (result["final_judgement"]==0).sum())
col3.metric("정상 기업 수", (result["final_judgement"]==1).sum())

# 🚨 리스크 기업만 보기
show_risk_only = st.checkbox("🚨 리스크 기업만 보기", value=False)
if show_risk_only:
    result = result[result["final_judgement"]==0]

# 📋 결과 테이블
st.subheader("기업별 예측 결과")
show_cols = ["사업자번호","업체명","p_risk","p_normal","final_judgement"]
for col in ["부족액_예상","부족액_실제"]:
    if col in result.columns:
        show_cols.append(col)

st.dataframe(result[show_cols])

# ===============================
# 4) 상세보기
# ===============================
if not result.empty:
    row = result.iloc[0]
    st.markdown("---")
    st.subheader(f"🏢 {row['업체명']} ({row['사업자번호']})")

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.metric("리스크 확률", f"{row['p_risk']:.1%}")
    with c2:
        st.metric("정상 확률", f"{row['p_normal']:.1%}")
    with c3:
        st.markdown("**최종 판정:**")
        if row["final_judgement"] == 0:
            st.error("🚨 리스크 (70%↑)")
        else:
            st.success("✅ 정상")

    # ===============================
    # 📉 부족액 비교 (예상 vs 실제)
    # ===============================
    if "부족액_예상" in row and "부족액_실제" in row:
        st.markdown("### 📉 부족액 비교 (예상 vs 실제)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("예상 부족액", f"{int(row['부족액_예상']):,} 원")
        with col2:
            st.metric("실제 부족액", f"{int(row['부족액_실제']):,} 원")

        diff = row["부족액_실제"] - row["부족액_예상"]
        if diff > 0:
            st.error(f"⚠️ 실제 부족액이 예상보다 {int(diff):,} 원 더 큼")
        elif diff < 0:
            st.success(f"✅ 실제 부족액이 예상보다 {abs(int(diff)):,} 원 적음")
        else:
            st.info("📊 실제와 예상 부족액이 동일합니다.")

        # ===============================
        # 🔄 부족액 기반 리밸런싱 + 월별 예상 잔고
        # ===============================
        if row["부족액_예상"] > 0:  # 🚩 예상 부족액 있을 때만
            st.markdown("---")
            st.markdown("### 🔄 원리금 / 비원리금 리밸런싱 + 월별 예상 잔고")
            st.write("가정: 원리금 보장 5%, 비원리금 10% (연이율, 1년 만기 기준)")

            total = row["부족액_예상"]

            scenarios = {
                "보수형 (원리금70%)": (total * 0.7, total * 0.3),
                "균형형 (50:50)": (total * 0.5, total * 0.5),
                "공격형 (비원리금70%)": (total * 0.3, total * 0.7),
            }

            months = list(range(1, 13))
            df_bal = pd.DataFrame({"n개월 후": months})

            for name, (principal_amt, non_principal_amt) in scenarios.items():
                balances = []
                for m in months:
                    bal = principal_amt * (1 + 0.05 / 12) ** m + non_principal_amt * (1 + 0.10 / 12) ** m
                    balances.append(f"{int(bal):,} 원")
                df_bal[name] = balances

            st.dataframe(df_bal, hide_index=True)

    # ===============================
    # 📌 설명 근거
    # ===============================
    if row["final_judgement"] == 0:
        st.markdown("### 📌 설명 근거")
        explanation_lines = str(row["explanation"]).split(" / ")
        for line in explanation_lines:
            if line.strip():
                st.markdown(f"- ✔ {line.strip()}")

    # ===============================
    # 5) 연도별 지표 트렌드 그래프
    # ===============================
    st.markdown("---")
    st.markdown("### 📈 연도별 지표 트렌드")

    @st.cache_data
    def load_raw_data():
        raw = pd.read_csv("퇴직연금_통합_데이터_2014_2024.csv")
        raw.columns = raw.columns.str.strip()
        raw["기준연도"] = raw["기준연도"].astype(str).str[:4].astype(int)
        return raw

    raw_df = load_raw_data()

    if "사업자번호" in row:
        firm_data = raw_df[raw_df["사업자번호"] == row["사업자번호"]].copy()
    else:
        firm_data = raw_df[raw_df["업체명"] == row["업체명"]].copy()

    if not firm_data.empty:
        firm_data["적립률"] = firm_data["적립금"] / firm_data["최소적립금(적립기준액)"]
        firm_data["준수비율"] = firm_data["평가적립금합계"] / firm_data["계속기준책임준비금"]
        firm_data["납입이행률"] = firm_data["부담금납입액"] / firm_data["부담금산정액"]

        chart_df = firm_data[["기준연도","적립률","준수비율","납입이행률"]].set_index("기준연도")

        st.line_chart(chart_df)
    else:
        st.warning("해당 기업의 원본 연도별 데이터가 없습니다.")
