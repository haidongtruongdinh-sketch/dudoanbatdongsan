# Chạy: pip install streamlit scikit-learn pandas numpy  →  streamlit run app.py
import zlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Định giá nhà TP.HCM", page_icon="🏠", layout="wide")

# ---------- Dữ liệu nền (triệu đồng / m², nhà phố, hẻm xe hơi, sổ hồng) ----------
DISTRICTS = {
    "Quận 1": 350, "Quận 3": 250, "Quận 5": 160, "Quận 10": 180, "Phú Nhuận": 170,
    "Quận 4": 130, "Bình Thạnh": 130, "Quận 7": 120, "Tân Bình": 120, "Thủ Đức": 95,
    "Quận 11": 100, "Quận 6": 95, "Gò Vấp": 85, "Tân Phú": 90, "Quận 8": 75,
    "Bình Tân": 70, "Quận 12": 65, "Nhà Bè": 60, "Bình Chánh": 45, "Hóc Môn": 40, "Củ Chi": 30,
}
TYPES = {"Căn hộ chung cư": 0.5, "Nhà phố": 1.0, "Biệt thự": 1.25, "Đất nền": 0.8}
AREA = {"Căn hộ chung cư": (35, 120), "Nhà phố": (25, 150), "Biệt thự": (150, 400), "Đất nền": (50, 300)}
FRONT = {"Mặt tiền đường": 1.35, "Hẻm xe hơi": 1.0, "Hẻm nhỏ": 0.8}
LEGAL = {"Sổ hồng / Sổ đỏ": 1.0, "Hợp đồng mua bán": 0.92, "Giấy tờ khác": 0.85}
ICON = {"Căn hộ chung cư": "🏢", "Nhà phố": "🏠", "Biệt thự": "🏡", "Đất nền": "🌳"}


def true_price(df):
    """Giá mô phỏng (tỷ đồng) dùng để tạo dữ liệu huấn luyện."""
    house = df.type.isin(["Nhà phố", "Biệt thự"])
    ppm = df.district.map(DISTRICTS) * df.type.map(TYPES) * df.legal.map(LEGAL)
    ppm *= np.where(df.type == "Căn hộ chung cư", 1, df.frontage.map(FRONT))
    ppm *= (df.area / 60) ** -0.1 * (1 + 0.015 * (df.bed - 2))
    ppm *= np.where(house, 1 + 0.04 * (df.floors - 1), 1)
    return ppm * df.area / 1000


def encode(df, cols=None):
    x = pd.get_dummies(df, columns=["district", "type", "frontage", "legal"])
    return x if cols is None else x.reindex(columns=cols, fill_value=0)


@st.cache_resource
def train():
    rng = np.random.default_rng(8)
    n = 6000
    t = rng.choice(list(TYPES), n, p=[.35, .4, .1, .15])
    lo = np.array([AREA[k][0] for k in t]); hi = np.array([AREA[k][1] for k in t])
    df = pd.DataFrame({
        "district": rng.choice(list(DISTRICTS), n), "type": t,
        "frontage": rng.choice(list(FRONT), n), "legal": rng.choice(list(LEGAL), n, p=[.7, .2, .1]),
        "area": rng.uniform(lo, hi), "bed": rng.integers(1, 6, n),
        "floors": np.where(np.isin(t, ["Nhà phố", "Biệt thự"]), rng.integers(1, 6, n), 1),
    })
    df["bath"] = np.clip(df.bed + rng.integers(-1, 2, n), 1, None)
    y = true_price(df) * rng.normal(1, 0.06, n)
    X = encode(df)
    model = RandomForestRegressor(120, min_samples_leaf=3, n_jobs=-1, random_state=8)
    model.fit(X.to_numpy(float), y)
    return model, list(X.columns)


def estimate(rows):
    model, cols = train()
    X = encode(pd.DataFrame(rows), cols).to_numpy(float)
    p = np.array([t.predict(X) for t in model.estimators_])
    return p.mean(0), np.percentile(p, 10, 0), np.percentile(p, 90, 0)


def fmt(ty):
    return f"{ty * 1000:,.0f} triệu" if ty < 1 else f"{ty:.2f} tỷ".replace(".", ",")


# ---------- Giao diện ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stApp * {font-family:'Be Vietnam Pro',sans-serif;}
.stApp {background:#F4F4F4; color:#222;}
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding:0 1.5rem 5rem; max-width:1200px;}
.topbar {background:#FFBA00; position:relative; left:50%; margin-left:-50vw; width:100vw; margin-bottom:24px;
  box-shadow:0 1px 0 rgba(0,0,0,.08);}
.topbar div {max-width:1200px; margin:auto; padding:14px 24px; display:flex; justify-content:space-between; align-items:center;}
.logo {font-size:24px; font-weight:800; color:#222;}
.tag {font-size:14px; font-weight:500; color:#222;}
.hero {font-size:28px; font-weight:800; margin:4px 0 4px;}
.sub {color:#666; margin-bottom:20px;}
[data-testid="stForm"] {background:#fff; border:0; border-radius:12px; padding:1.4rem 1.4rem .8rem;
  box-shadow:0 1px 4px rgba(0,0,0,.08);}
.stFormSubmitButton button, .stButton button {background:#FFBA00; color:#222; border:0; font-weight:700;
  border-radius:8px; width:100%; padding:.6rem 0;}
.stFormSubmitButton button:hover {background:#E6A700; color:#222;}
.stTabs [data-baseweb="tab"] {font-weight:600; font-size:16px;}
.stTabs [data-baseweb="tab-highlight"] {background:#FFBA00; height:3px;}
.card {background:#fff; border-radius:12px; padding:22px 24px; box-shadow:0 1px 4px rgba(0,0,0,.08); margin-bottom:16px;}
.label {color:#666; font-size:14px;}
.big {color:#E03C31; font-size:44px; font-weight:800; line-height:1.15; margin:2px 0 6px;}
.stats {display:flex; gap:12px; margin-top:16px; flex-wrap:wrap;}
.stat {flex:1; min-width:120px; background:#F4F4F4; border-radius:8px; padding:10px 14px;}
.stat b {display:block; font-size:18px;}
.empty {text-align:center; color:#666; padding:48px 20px;}
.grid {display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px;}
.item {background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.08);}
.thumb {height:140px; display:flex; align-items:center; justify-content:center; font-size:52px;
  background:linear-gradient(135deg,#FFE7A3,#FFBA00);}
.item .body {padding:12px 14px 14px;}
.item .t {font-weight:600; margin-bottom:6px;}
.item .p {color:#E03C31; font-weight:800; font-size:18px;}
.item .m {color:#666; font-size:13px; margin-top:4px;}
.note {color:#888; font-size:12.5px; margin-top:8px;}
.wm {position:fixed; left:16px; bottom:10px; z-index:9999; font-size:13px; font-weight:700;
  letter-spacing:.08em; color:#222; opacity:.28; pointer-events:none; user-select:none;}
</style>
<div class="topbar"><div><span class="logo">🏠 Định Giá Nhà</span><span class="tag">Bất động sản TP. Hồ Chí Minh</span></div></div>
<div class="wm">NHÓM THỰC HIỆN 8</div>
<div class="hero">Dự đoán giá bất động sản trong vài giây</div>
<div class="sub">Nhập thông tin căn nhà, hệ thống ước tính giá bán và khoảng giá hợp lý.</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Dự đoán giá", "Tin tham khảo"])

with tab1:
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.form("predict"):
            c1, c2 = st.columns(2)
            ptype = c1.selectbox("Loại hình", list(TYPES), index=1)
            district = c2.selectbox("Quận / Huyện", list(DISTRICTS), index=list(DISTRICTS).index("Bình Thạnh"))
            c3, c4 = st.columns(2)
            area = c3.number_input("Diện tích (m²)", 10.0, 1000.0, 60.0, 5.0)
            floors = c4.number_input("Số tầng", 1, 10, 2)
            c5, c6 = st.columns(2)
            bed = c5.number_input("Phòng ngủ", 1, 10, 2)
            bath = c6.number_input("Phòng tắm", 1, 10, 2)
            c7, c8 = st.columns(2)
            front = c7.selectbox("Vị trí", list(FRONT), index=1)
            legal = c8.selectbox("Giấy tờ", list(LEGAL))
            go = st.form_submit_button("Dự đoán giá")
        if go:
            base = dict(district=district, type=ptype, frontage=front, legal=legal,
                        area=area, bed=bed, bath=bath, floors=floors)
            m, lo_, hi_ = estimate([base])
            st.session_state["res"] = (base, m[0], lo_[0], hi_[0])

    with right:
        if "res" not in st.session_state:
            st.markdown('<div class="card empty">🏠<br><br><b>Chưa có kết quả</b><br>'
                        'Nhập thông tin bên trái rồi bấm “Dự đoán giá”.</div>', unsafe_allow_html=True)
        else:
            base, price, lo_, hi_ = st.session_state["res"]
            st.markdown(
                f'<div class="card"><div class="label">Giá ước tính · {base["type"]} · {base["district"]}</div>'
                f'<div class="big">{fmt(price)}</div>'
                f'<div class="label">Khoảng hợp lý: {fmt(lo_)} – {fmt(hi_)}</div>'
                f'<div class="stats">'
                f'<div class="stat"><span class="label">Giá / m²</span><b>{price * 1000 / base["area"]:,.0f} triệu</b></div>'
                f'<div class="stat"><span class="label">Diện tích</span><b>{base["area"]:,.0f} m²</b></div>'
                f'<div class="stat"><span class="label">Phòng ngủ</span><b>{base["bed"]} PN · {base["bath"]} WC</b></div>'
                f'</div></div>', unsafe_allow_html=True)
            rows = [{**base, "district": d} for d in DISTRICTS]
            mean, _, _ = estimate(rows)
            cmp = pd.DataFrame({"Quận": list(DISTRICTS), "Giá/m² (triệu)": mean * 1000 / base["area"]})
            st.markdown('<div class="card"><b>8 khu vực giá/m² cao nhất</b> '
                        '<span class="label">(cùng cấu hình căn nhà của bạn)</span></div>', unsafe_allow_html=True)
            st.bar_chart(cmp.nlargest(8, "Giá/m² (triệu)"), x="Quận", y="Giá/m² (triệu)",
                         color="#FFBA00", horizontal=True)

with tab2:
    f1, f2 = st.columns(2)
    d2 = f1.selectbox("Khu vực", list(DISTRICTS), index=list(DISTRICTS).index("Quận 7"), key="d2")
    t2 = f2.selectbox("Loại hình", list(TYPES), index=0, key="t2")
    rng = np.random.default_rng(zlib.crc32(f"{d2}{t2}".encode()))
    lo, hi = AREA[t2]
    rows = []
    for _ in range(6):
        house = t2 in ("Nhà phố", "Biệt thự")
        b = int(rng.integers(1, 5))
        rows.append(dict(district=d2, type=t2, frontage=str(rng.choice(list(FRONT))),
                         legal=str(rng.choice(list(LEGAL), p=[.7, .2, .1])), area=float(rng.uniform(lo, hi)),
                         bed=b, bath=b, floors=int(rng.integers(1, 5)) if house else 1))
    prices, _, _ = estimate(rows)
    cards = "".join(
        f'<div class="item"><div class="thumb">{ICON[t2]}</div><div class="body">'
        f'<div class="p">{fmt(p)}</div><div class="t">{t2} {r["area"]:.0f} m² · {r["bed"]} PN</div>'
        f'<div class="m">{r["frontage"]} · {r["legal"]}</div><div class="m">📍 {d2}, TP.HCM</div></div></div>'
        for r, p in zip(rows, prices))
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)

st.markdown('<div class="note">Mô hình Random Forest huấn luyện trên dữ liệu mô phỏng, kết quả chỉ mang tính tham khảo. '
            'Thay bằng dữ liệu thật (CSV) trong hàm <code>train()</code> để tăng độ chính xác.</div>',
            unsafe_allow_html=True)
