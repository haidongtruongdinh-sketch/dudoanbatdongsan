# Chạy: pip install streamlit scikit-learn pandas numpy requests  →  streamlit run app.py
import html
import json
import re
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Dự đoán giá nhà TP.HCM", page_icon="🏠", layout="wide")

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


DEF_AREA = {"Căn hộ chung cư": 70, "Nhà phố": 60, "Biệt thự": 250, "Đất nền": 100}


def true_price(df):
    """Giá mô phỏng (tỷ đồng). Đất nền không tính phòng/tầng, căn hộ không tính tầng/hẻm."""
    t = df.type
    ppm = df.district.map(DISTRICTS) * t.map(TYPES) * df.legal.map(LEGAL)
    ppm *= np.where(t == "Căn hộ chung cư", 1, df.frontage.map(FRONT))
    ppm *= (df.area / 60) ** -0.1
    ppm *= np.where(t == "Đất nền", 1, 1 + 0.015 * (df.bed - 2))
    ppm *= np.where(t.isin(["Nhà phố", "Biệt thự"]), 1 + 0.04 * (df.floors - 1), 1)
    return ppm * df.area / 1000


def encode(df, cols=None):
    x = pd.get_dummies(df, columns=["district", "type", "frontage", "legal"])
    return x if cols is None else x.reindex(columns=cols, fill_value=0)


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    t = rng.choice(list(TYPES), n, p=[.35, .4, .1, .15])
    lo = np.array([AREA[k][0] for k in t]); hi = np.array([AREA[k][1] for k in t])
    df = pd.DataFrame({
        "district": rng.choice(list(DISTRICTS), n), "type": t,
        "frontage": rng.choice(list(FRONT), n), "legal": rng.choice(list(LEGAL), n, p=[.7, .2, .1]),
        "area": rng.uniform(lo, hi).round(0), "bed": rng.integers(1, 6, n),
        "floors": np.where(np.isin(t, ["Nhà phố", "Biệt thự"]), rng.integers(1, 6, n), 0),
    })
    df["bath"] = np.clip(df.bed + rng.integers(-1, 2, n), 1, None)
    df.loc[df.type == "Đất nền", ["bed", "bath"]] = 0
    return df, rng


@st.cache_resource
def train():
    df, rng = make_data(6000, 8)
    y = true_price(df) * rng.normal(1, 0.06, len(df))
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
.item a {color:#0B6BCB; text-decoration:none;}
.note {color:#888; font-size:12.5px; margin-top:8px;}
.wm {position:fixed; left:16px; bottom:10px; z-index:9999; font-size:13px; font-weight:700;
  letter-spacing:.08em; color:#222; opacity:.28; pointer-events:none; user-select:none;}
.thumb {position:relative;}
.badge {position:absolute; top:10px; left:10px; background:#2E9E5B; color:#fff; font-size:12px;
  font-weight:700; padding:3px 9px; border-radius:6px;}
</style>
<div class="topbar"><div><span class="logo">🏠 Dự Đoán Giá Nhà</span><span class="tag">Mua bán bất động sản TP. Hồ Chí Minh</span></div></div>
<div class="wm">NHÓM THỰC HIỆN 8</div>
<div class="hero">Dự đoán giá nhà / bất động sản tại TP.HCM</div>
<div class="sub">Nhập thông tin căn nhà, hệ thống ước tính giá bán và khoảng giá hợp lý.</div>
""", unsafe_allow_html=True)

STREETS = {
    "Quận 1": ["Hai Bà Trưng", "Lê Lợi"], "Quận 3": ["Võ Văn Tần", "Nam Kỳ Khởi Nghĩa"],
    "Quận 5": ["Trần Hưng Đạo", "An Dương Vương"], "Quận 10": ["Sư Vạn Hạnh", "Tô Hiến Thành"],
    "Phú Nhuận": ["Phan Xích Long", "Huỳnh Văn Bánh"], "Quận 4": ["Khánh Hội", "Hoàng Diệu"],
    "Bình Thạnh": ["Xô Viết Nghệ Tĩnh", "Điện Biên Phủ"], "Quận 7": ["Nguyễn Thị Thập", "Huỳnh Tấn Phát"],
    "Tân Bình": ["Cộng Hòa", "Hoàng Văn Thụ"], "Thủ Đức": ["Võ Văn Ngân", "Phạm Văn Đồng"],
    "Quận 11": ["Lạc Long Quân", "Minh Phụng"], "Quận 6": ["Hậu Giang", "Kinh Dương Vương"],
    "Gò Vấp": ["Quang Trung", "Lê Đức Thọ"], "Tân Phú": ["Tân Kỳ Tân Quý", "Lũy Bán Bích"],
    "Quận 8": ["Phạm Thế Hiển", "Tạ Quang Bửu"], "Bình Tân": ["Tên Lửa", "Mã Lò"],
    "Quận 12": ["Nguyễn Ảnh Thủ", "Hà Huy Giáp"], "Nhà Bè": ["Lê Văn Lương", "Nguyễn Bình"],
    "Bình Chánh": ["Trần Đại Nghĩa", "Quốc Lộ 50"], "Hóc Môn": ["Quốc Lộ 22", "Lê Thị Hà"],
    "Củ Chi": ["Tỉnh Lộ 8", "Nguyễn Thị Rành"],
}
PROJECTS = ["Sunrise Garden", "Green Valley", "Riverside Park", "Skyline Tower",
            "Lotus Residence", "Golden Land", "Ocean View", "Moonlight Plaza"]


@st.cache_data
def market():
    """Tin đăng mẫu tự tạo (chỉ dùng khi không tải được dữ liệu từ Nhatot)."""
    df, rng = make_data(300, 88)
    fair = estimate(df)[0]
    df["price"] = (true_price(df) * rng.normal(1, 0.1, len(df))).round(2)
    df["good"] = df.price <= fair * 0.95
    df["age"] = rng.integers(0, 30, len(df))
    addr, proj = [], []
    for r in df.itertuples():
        apt = r.type == "Căn hộ chung cư"
        hem = f"Hẻm {int(rng.integers(1, 90))}/" if not apt and r.frontage != "Mặt tiền đường" else ""
        addr.append(f"{hem}{int(rng.integers(1, 300))} {rng.choice(STREETS[r.district])}")
        proj.append(str(rng.choice(PROJECTS)) if apt else "")
    df["address"], df["project"] = addr, proj
    df["ppm"] = df.price * 1000 / df.area
    df["title"], df["url"], df["prj"] = "", "", False
    return df


# ===== BEGIN NHATOT: tự động lấy dữ liệu từ Nhatot, không cần upload file =====
NHATOT_URL = "https://www.nhatot.com/du-an-tp-ho-chi-minh"
CHOTOT_API = "https://gateway.chotot.com/v1/public/ad-listing"   # dự phòng: API tin đăng công khai
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
    "Referer": "https://www.nhatot.com/",
}
COLS = ["title", "district", "type", "frontage", "legal", "area", "bed", "bath", "floors",
        "price", "ppm", "age", "address", "project", "url", "prj", "good"]

# Tên khoá có thể gặp trong JSON (đã bỏ dấu _ và viết thường để so khớp)
K_NAME = ("name", "projectname", "title", "subject")
K_PRICE = ("price", "pricemin", "minprice", "fromprice", "pricefrom", "lowestprice",
           "pricestring", "pricedisplay", "pricetext", "priceformatted")
K_PPM = ("pricem2", "pricepersqm", "pricepermeter", "pricemillionperm2", "pricem2min",
         "unitprice", "avgpricem2", "averagepricem2")
K_AREA = ("size", "areasize", "acreage", "dientich", "areamin", "minarea", "sizemin",
          "landarea", "usablearea")
K_TYPE = ("categoryname", "category", "type", "typename", "producttype", "propertytype",
          "projecttype", "projectcategory")
K_ADDR = ("address", "fulladdress", "addressdisplay", "addresstext", "location", "street")
K_DIST = ("districtname", "district", "areaname", "districtfull")
K_WARD = ("wardname", "ward")
K_CITY = ("regionname", "cityname", "city", "provincename", "province")
K_URL = ("url", "link", "href", "permalink")


def _key(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _flat(d):
    """Gộp khoá cấp 1 + khoá của dict con (khoá cấp 1 được ưu tiên)."""
    n = {_key(k): v for k, v in d.items()}
    for v in d.values():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                n.setdefault(_key(k2), v2)
    return n


def _nums(s):
    """Lấy các số trong chuỗi kiểu Việt Nam: '1,56' → 1.56 ; '1.560.000.000' → 1.56e9."""
    out = []
    for m in re.findall(r"\d+(?:[.,]\d+)*", s):
        if "." in m and "," in m:
            m = m.replace(".", "").replace(",", ".") if m.rfind(",") > m.rfind(".") else m.replace(",", "")
        elif "," in m:
            m = m.replace(",", "") if re.fullmatch(r"[1-9]\d{0,2}(,\d{3})+", m) else m.replace(",", ".")
        elif "." in m and re.fullmatch(r"[1-9]\d{0,2}(\.\d{3})+", m):
            m = m.replace(".", "")
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def to_ty(x):
    """Giá → tỷ đồng (NaN nếu không hiểu). Khoảng giá thì lấy cận dưới."""
    if _isnum(x):
        return x / 1e9 if x >= 1e6 else np.nan
    if isinstance(x, str):
        t = x.lower()
        if "/m" in t:                      # đây là giá/m², không phải giá tổng
            return np.nan
        m = re.search(r"([\d.,]+)\s*(?:tỷ|tỉ)\s*(\d+)\s*(?:triệu|tr)\b", t)
        if m and _nums(m.group(1)):
            return _nums(m.group(1))[0] + float(m.group(2)) / 1000
        n = _nums(t)
        if not n:
            return np.nan
        if "tỷ" in t or "tỉ" in t:
            return n[0]
        if "triệu" in t or re.search(r"\btr\b", t):
            return n[0] / 1000
        return n[0] / 1e9 if n[0] >= 1e6 else np.nan
    return np.nan


def to_ppm(x):
    """Giá/m² → triệu đồng/m². Khoảng giá thì lấy trung bình."""
    if _isnum(x):
        return x / 1e6 if x >= 1e6 else (float(x) if x > 0 else np.nan)
    if isinstance(x, str):
        n = _nums(x)
        if not n:
            return np.nan
        v, t = float(np.mean(n[:2])), x.lower()
        if "tỷ" in t or "tỉ" in t:
            v *= 1000
        elif "nghìn" in t or "ngàn" in t:
            v /= 1000
        elif v >= 1e6:
            v /= 1e6
        return v
    return np.nan


def to_area(x):
    """Diện tích → m². Khoảng thì lấy trung bình."""
    if _isnum(x):
        return float(x) if 0 < x < 1e5 else np.nan
    if isinstance(x, str):
        n = _nums(x)
        return float(np.mean(n[:2])) if n and 0 < n[0] < 1e5 else np.nan
    return np.nan


def guess_type(text):
    t = text.lower()
    if re.search(r"nhà phố|liền kề|shophouse|townhouse|nhà riêng|nhà ở|nhà mặt phố|nhà hẻm", t):
        return "Nhà phố"
    if re.search(r"biệt thự|villa|dinh thự", t):
        return "Biệt thự"
    if re.search(r"căn hộ|chung cư|apartment|condotel|officetel", t):
        return "Căn hộ chung cư"
    if re.search(r"đất nền|đất thổ cư|đất ở|bán đất|land", t):
        return "Đất nền"
    return "Khác"


def find_district(text):
    t = text.lower()
    for d in sorted(DISTRICTS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(d.lower())}(?!\w)", t):
            return d
    return None


def _looks(d):
    n = _flat(d)
    has = lambda ks: any(n.get(k) not in (None, "", [], {}) for k in ks)
    return has(K_NAME) and (has(K_PRICE) or has(K_PPM))


def _find(o, out, depth=0):
    """Duyệt JSON, gom mọi phần tử trông giống 1 dự án / 1 tin đăng."""
    if depth > 14:
        return
    if isinstance(o, dict):
        for v in o.values():
            _find(v, out, depth + 1)
    elif isinstance(o, list):
        for x in o:
            if isinstance(x, dict) and _looks(x):
                out.append(x)
            else:
                _find(x, out, depth + 1)


def to_row(d, is_prj, fixed_type=None):
    n = _flat(d)

    def pick(keys):
        for k in keys:
            v = n.get(k)
            if isinstance(v, dict):
                v = next((v[s] for s in ("min", "from", "value", "text", "display", "formatted")
                          if v.get(s) not in (None, "")), None)
            if (isinstance(v, str) and v.strip()) or _isnum(v):
                return v
        return None

    def num(*keys):
        try:
            return int(float(pick(keys)))
        except (TypeError, ValueError):
            return 0

    name = str(pick(K_NAME) or "").strip()
    raw_p, raw_m = pick(K_PRICE), pick(K_PPM)
    if isinstance(raw_p, str) and "/m" in raw_p.lower():      # chuỗi giá/m² nằm nhầm trong khoá giá
        raw_p, raw_m = None, raw_m or raw_p
    price, ppm = to_ty(raw_p), to_ppm(raw_m)
    raw_a = pick(K_AREA)
    if raw_a is None and isinstance(n.get("area"), str) and "m" in n["area"].lower():
        raw_a = n["area"]
    area = to_area(raw_a)
    if np.isnan(ppm) and not np.isnan(price) and not np.isnan(area):
        ppm = price * 1000 / area
    if np.isnan(price) and not np.isnan(ppm) and not np.isnan(area):
        price = ppm * area / 1000
    if np.isnan(price) and np.isnan(ppm):
        return None
    if (not np.isnan(ppm) and not 1 <= ppm <= 2000) or (not np.isnan(price) and not 0.05 <= price <= 3000):
        return None                                           # loại số vô lý

    ward, dist_raw, city, addr = pick(K_WARD), pick(K_DIST), pick(K_CITY), pick(K_ADDR)
    parts = [str(x).strip() for x in (addr, ward, dist_raw) if x not in (None, "")]
    address = ", ".join(dict.fromkeys(parts))
    if not name or not address:
        return None                                           # bỏ các mục lọc/menu không phải tin
    district = (find_district(" ".join(parts + [str(city or "")]))
                or (str(dist_raw).strip() if dist_raw else None)
                or (str(ward).strip() if ward else None) or "Khác")

    typ = fixed_type or guess_type(" ".join(str(pick((k,)) or "") for k in K_TYPE))
    if typ == "Khác":
        typ = guess_type(name)
    ht = str(n.get("housetype", ""))                          # mã loại nhà của API tin đăng
    typ = "Biệt thự" if ht == "3" else ("Nhà phố" if ht == "4" else typ)
    frontage = {"1": "Mặt tiền đường", "2": "Hẻm xe hơi"}.get(ht, "")
    lg = str(n.get("propertylegaldocument", ""))
    legal = {"1": "Sổ hồng / Sổ đỏ", "2": "Hợp đồng mua bán", "3": "Giấy tờ khác"}.get(lg, "")

    lt = pick(("listtime",))
    age = max(0, int((time.time() * 1000 - float(lt)) // 86_400_000)) if _isnum(lt) and lt > 1e11 else np.nan
    url = pick(K_URL)
    url = url if isinstance(url, str) and url.startswith("http") else (
        "https://www.nhatot.com" + url if isinstance(url, str) and url.startswith("/") else "")
    return dict(title=name, district=district, type=typ, frontage=frontage, legal=legal, area=area,
                bed=num("rooms", "bedroom", "bedrooms", "beds"), bath=num("toilets", "bathroom", "bathrooms", "baths"),
                floors=0 if typ in ("Căn hộ chung cư", "Đất nền") else num("floors", "numberoffloors"),
                price=price, ppm=ppm, age=age, address=address, project=name if is_prj else "",
                url=url, prj=is_prj, good=False)


def _next_data(text):
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', text, re.S)
    return json.loads(m.group(1)) if m else None


@st.cache_data(ttl=900, show_spinner="Đang tải dữ liệu từ Nhatot...")
def load_nhatot(max_pages=5):
    """Trả về (DataFrame | None, nhật ký, nhãn nguồn). Không bao giờ ném lỗi ra ngoài."""
    log, rows, seen = [], [], set()
    s = requests.Session()
    s.headers.update(HEADERS)

    def add(item, is_prj, fixed_type=None):
        r = to_row(item, is_prj, fixed_type)
        k = (r["title"], r["address"], r["price"], r["ppm"]) if r else None
        if r and k not in seen:
            seen.add(k)
            rows.append(r)
            return True
        return False

    # 1) Trang dự án mà bạn chỉ định: đọc JSON nhúng sẵn trong HTML (__NEXT_DATA__)
    for p in range(1, max_pages + 1):
        url = NHATOT_URL if p == 1 else f"{NHATOT_URL}?page={p}"
        try:
            r = s.get(url, timeout=20)
            r.encoding = "utf-8"
            log.append(f"Trang dự án {p}: HTTP {r.status_code}")
            if r.status_code != 200:
                break
            data = _next_data(r.text)
            if data is None:
                log.append("  Không thấy __NEXT_DATA__ (trang có thể đang chặn bot hoặc đổi cấu trúc)")
                break
            items = []
            _find(data, items)
            got = sum(add(x, True) for x in items)
            log.append(f"  Tìm thấy {len(items)} mục, đọc được {got} mục mới")
            if got == 0:
                break
        except Exception as e:
            log.append(f"Trang dự án {p}: lỗi {type(e).__name__}")
            break
    label = "Nhatot.com – trang Dự án TP.HCM (giá hiển thị là giá khởi điểm)"

    # 2) Dự phòng: API tin đăng công khai (căn hộ / nhà ở / đất tại TP.HCM)
    if not rows:
        log.append("Trang dự án không cho dữ liệu → thử API tin đăng của Chotot/Nhatot")
        label = "Nhatot.com – tin đăng mua bán tại TP.HCM"
        for cg, typ in ((1010, "Căn hộ chung cư"), (1020, "Nhà phố"), (1040, "Đất nền")):
            for page in range(3):
                try:
                    r = s.get(CHOTOT_API, timeout=20, params=dict(
                        cg=cg, region_v2=13000, limit=50, o=page * 50, page=page + 1, st="s,k"))
                    ads = r.json().get("ads", []) if r.status_code == 200 else []
                    log.append(f"API cg={cg} trang {page + 1}: HTTP {r.status_code}, {len(ads)} tin")
                    got = sum(add(a, False, typ) for a in ads)
                    if not ads or got == 0:
                        break
                except Exception as e:
                    log.append(f"API cg={cg}: lỗi {type(e).__name__}")
                    break

    if not rows:
        log.append("→ Không lấy được dữ liệu thật, app dùng dữ liệu mẫu.")
        return None, log, ""
    df = pd.DataFrame(rows, columns=COLS)
    log.append(f"→ Tổng cộng {len(df)} tin.")
    return df, log, label
# ===== END NHATOT =====


def card(r):
    e = html.escape                       # tránh chèn HTML lạ từ dữ liệu web
    apt = r.type == "Căn hộ chung cư"
    title = e(r.title or f"{r.type} {r.project}".strip())
    meta = ([e(r.type)] if r.title and r.type != "Khác" else []) + ([f"{r.area:.0f} m²"] if pd.notna(r.area) else [])
    meta += ([f"{r.bed} PN · {r.bath} WC"] if r.bed else []) + ([f"{r.floors} tầng"] if r.floors else [])
    where = " · ".join(x for x in ([] if apt else [r.frontage]) + [r.legal] if x)
    ago = "" if pd.isna(r.age) else ("Hôm nay" if r.age == 0 else f"{int(r.age)} ngày trước")
    where = e(" · ".join(x for x in (where, ago) if x))
    badge = '<span class="badge">Giá tốt</span>' if r.good else ""
    loc = e(", ".join(x for x in (r.address, r.district if r.district not in r.address and r.district != "Khác" else "") if x))
    price = (("Từ " if r.prj else "") + fmt(r.price)) if pd.notna(r.price) else "Liên hệ"
    ppm = f' <span class="m">· {r.ppm:,.0f} tr/m²</span>' if pd.notna(r.ppm) else ""
    link = f'<div class="m"><a href="{e(r.url, quote=True)}" target="_blank">Xem tin gốc ↗</a></div>' if r.url else ""
    return (f'<div class="item"><div class="thumb">{ICON.get(r.type, "🏠")}{badge}</div><div class="body">'
            f'<div class="p">{price}{ppm}</div>'
            f'<div class="t">{title}</div><div class="m">{" · ".join(meta)}</div>'
            + (f'<div class="m">{where}</div>' if where else "")
            + f'<div class="m">📍 {loc}, TP.HCM</div>{link}</div></div>')


tab1, tab2 = st.tabs(["Dự đoán giá", "Nhà mua bán"])

with tab1:
    left, right = st.columns([5, 6], gap="large")
    with left:
        ptype = st.selectbox("Loại hình", list(TYPES), index=1)
        apt, land = ptype == "Căn hộ chung cư", ptype == "Đất nền"
        house = not apt and not land
        if land:
            st.caption("Đất nền chưa xây dựng nên không có phòng ngủ, phòng tắm hay số tầng.")
        elif apt:
            st.caption("Căn hộ được tính theo diện tích, số phòng và giấy tờ, không tính số tầng hay hẻm.")
        with st.form("predict"):
            c1, c2 = st.columns(2)
            district = c1.selectbox("Quận / Huyện", list(DISTRICTS), index=list(DISTRICTS).index("Bình Thạnh"))
            area = c2.number_input("Diện tích (m²)" if apt else "Diện tích đất (m²)",
                                   10.0, 5000.0, float(DEF_AREA[ptype]), 5.0)
            bed = bath = floors = 0
            if not land:
                c3, c4 = st.columns(2)
                bed = c3.number_input("Phòng ngủ", 1, 10, 2)
                bath = c4.number_input("Phòng tắm", 1, 10, 2)
            if house:
                floors = st.number_input("Số tầng", 1, 10, 2)
            if apt:
                front = "Hẻm xe hơi"
                legal = st.selectbox("Giấy tờ", list(LEGAL))
            else:
                c5, c6 = st.columns(2)
                front = c5.selectbox("Vị trí", list(FRONT), index=1)
                legal = c6.selectbox("Giấy tờ", list(LEGAL))
            go = st.form_submit_button("Dự đoán giá")
        if go:
            base = dict(district=district, type=ptype, frontage=front, legal=legal,
                        area=area, bed=bed, bath=bath, floors=floors)
            m, lo_, hi_ = estimate([base])
            st.session_state["res"] = (base, m[0], lo_[0], hi_[0])

    with right:
        if "res" not in st.session_state:
            st.markdown('<div class="card empty">🏠<br><br><b>Chưa có kết quả</b><br>'
                        'Chọn loại hình, nhập thông tin rồi bấm “Dự đoán giá”.</div>', unsafe_allow_html=True)
        else:
            base, price, lo_, hi_ = st.session_state["res"]
            k, v = ("Phòng ngủ", f'{base["bed"]} PN · {base["bath"]} WC') if base["bed"] else ("Vị trí", base["frontage"])
            st.markdown(
                f'<div class="card"><div class="label">Giá ước tính · {base["type"]} · {base["district"]}</div>'
                f'<div class="big">{fmt(price)}</div>'
                f'<div class="label">Khoảng hợp lý: {fmt(lo_)} – {fmt(hi_)}</div>'
                f'<div class="stats">'
                f'<div class="stat"><span class="label">Giá / m²</span><b>{price * 1000 / base["area"]:,.0f} triệu</b></div>'
                f'<div class="stat"><span class="label">Diện tích</span><b>{base["area"]:,.0f} m²</b></div>'
                f'<div class="stat"><span class="label">{k}</span><b>{v}</b></div>'
                f'</div></div>', unsafe_allow_html=True)
            rows = [{**base, "district": d} for d in DISTRICTS]
            mean, _, _ = estimate(rows)
            cmp = pd.DataFrame({"Quận": list(DISTRICTS), "Giá/m² (triệu)": mean * 1000 / base["area"]})
            st.markdown('<div class="card"><b>8 khu vực giá/m² cao nhất</b> '
                        '<span class="label">(cùng cấu hình bất động sản của bạn)</span></div>', unsafe_allow_html=True)
            st.bar_chart(cmp.nlargest(8, "Giá/m² (triệu)"), x="Quận", y="Giá/m² (triệu)",
                         color="#FFBA00", horizontal=True)

with tab2:
    real, log, label = load_nhatot()
    mk = real if real is not None and not real.empty else market()
    if real is not None and not real.empty:
        st.caption(f"Nguồn: {label} · {len(mk)} tin, tự động cập nhật mỗi 15 phút.")
    else:
        st.warning("Chưa tải được dữ liệu từ Nhatot nên đang hiển thị dữ liệu mẫu tự tạo. "
                   "Mở “Trạng thái nguồn dữ liệu” bên dưới để xem lý do.")
    with st.expander("Trạng thái nguồn dữ liệu"):
        st.markdown("\n".join(f"- {x}" for x in log) or "- (chưa có nhật ký)")
        if st.button("Tải lại dữ liệu từ Nhatot"):
            st.cache_data.clear()
            st.rerun()

    amax = int(max(400, np.ceil(np.nan_to_num(mk.area.max(), nan=0) / 5) * 5))
    r1 = st.columns(3)
    dsel = r1[0].multiselect("Khu vực", sorted(mk.district.unique()), placeholder="Tất cả khu vực")
    tsel = r1[1].multiselect("Loại hình", sorted(mk.type.unique()), placeholder="Tất cả loại hình")
    SORT = {"Mới đăng": ("age", True), "Giá thấp đến cao": ("price", True),
            "Giá cao đến thấp": ("price", False), "Diện tích lớn nhất": ("area", False)}
    sort = r1[2].selectbox("Sắp xếp", list(SORT))
    r2 = st.columns(3)
    blo, bhi = r2[0].slider("Ngân sách (tỷ đồng)", 0.0, 50.0, (0.0, 50.0), 0.5)
    alo, ahi = r2[1].slider("Diện tích (m²)", 10, amax, (10, amax), 5)
    beds = r2[2].selectbox("Phòng ngủ", [0, 1, 2, 3, 4], format_func=lambda x: "Tất cả" if x == 0 else f"Từ {x} PN")

    f = mk
    if dsel:
        f = f[f.district.isin(dsel)]
    if tsel:
        f = f[f.type.isin(tsel)]
    # tin chưa có giá / diện tích vẫn hiện khi bộ lọc tương ứng đang để toàn khoảng
    price_ok = ((f.price >= blo) & ((f.price <= bhi) | (bhi >= 50))) | (f.price.isna() & (blo <= 0) & (bhi >= 50))
    area_ok = f.area.between(alo, ahi) | (f.area.isna() & (alo <= 10) & (ahi >= amax))
    f = f[price_ok & area_ok]
    if beds:
        f = f[f.bed >= beds]
    col, asc = SORT[sort]
    f = f.sort_values(col, ascending=asc)

    shown = st.session_state.get("shown", 12)
    st.markdown(f'<div class="label" style="margin:8px 0 12px">Tìm thấy <b>{len(f)}</b> tin phù hợp</div>',
                unsafe_allow_html=True)
    if f.empty:
        st.markdown('<div class="card empty"><b>Không có căn nào phù hợp</b><br>'
                    'Thử tăng ngân sách hoặc chọn thêm khu vực.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="grid">' + "".join(card(r) for r in f.head(shown).itertuples()) + '</div>',
                    unsafe_allow_html=True)
        if len(f) > shown and st.button("Xem thêm"):
            st.session_state["shown"] = shown + 12
            st.rerun()

st.markdown('<div class="note">Mô hình Random Forest huấn luyện trên dữ liệu mô phỏng. Mục Nhà mua bán lấy dữ liệu '
            'từ Nhatot khi tải được, nếu không sẽ dùng dữ liệu mẫu. Kết quả chỉ mang tính tham khảo.</div>',
            unsafe_allow_html=True)
