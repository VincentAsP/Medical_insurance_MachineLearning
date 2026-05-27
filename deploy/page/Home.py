import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler

# --- 1. INISIALISASI SESSION STATE ---
if "inputan" not in st.session_state:
    st.session_state.inputan = 1
    
if "jawaban_user" not in st.session_state:
    st.session_state.jawaban_user = {}

# Fungsi Navigasi
def lanjut_step(nama_fitur):
    st.session_state.jawaban_user[nama_fitur] = st.session_state[nama_fitur]
    st.session_state.inputan += 1

def undo():
    if st.session_state.inputan > 1:
        st.session_state.inputan -= 1

def reset():
    st.session_state.clear()
    st.session_state.inputan = 1
    st.session_state.jawaban_user = {}

# --- 2. CACHE PREPROCESSING (SCALER & ENCODER) ---
@st.cache_resource
def setup_preprocessing():
    try:
        df = pd.read_csv('deploy/medical_insurance.csv')
    except:
        df = pd.read_csv('medical_insurance.csv')
        
    df['alcohol_freq'] = df['alcohol_freq'].fillna('Unknown')
    
    cat_cols = ['sex', 'smoker', 'alcohol_freq']
    encoders = {}
    categories = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        categories[col] = list(le.classes_)
        
    feature_cols = [
        'age', 'sex', 'bmi', 'smoker', 'alcohol_freq',
        'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c',
        'visits_last_year', 'hospitalizations_last_3yrs',
        'days_hospitalized_last_3yrs', 'medication_count',
        'proc_imaging_count', 'proc_surgery_count', 'proc_physio_count',
        'proc_consult_count', 'proc_lab_count',
    ]
    
    X = df[feature_cols]
    scaler = RobustScaler()
    scaler.fit(X)
    
    return scaler, encoders, feature_cols, categories

scaler_obj, encoders_dict, feature_cols, categories_dict = setup_preprocessing()

# --- 3. CACHE MODEL ML ---
@st.cache_resource
def load_all_models():
    return {
        "XGBoost": joblib.load('deploy/XGB.pkl'),
        "Logistic Regression": joblib.load('deploy/LR.pkl'),
        "Random Forest": joblib.load('deploy/RF.pkl'),
        "Decision Tree": joblib.load('deploy/DT.pkl')
    }

pilihan_model_dict = load_all_models()

# --- 4. BAGIAN UI HEADER ---
st.title("Medical Insurance 🏥")
st.write("""
**Kelompok:**
1. Aretha Tiurma S.M - 2802468220
2. Muhammad Edgar T.S - 2802545352
3. Vincentius Axel S.P - 2802528554
""")

st.markdown("<h2 style='color:cyan'>Prediksi & Komparasi Risiko Kesehatan</h2>", unsafe_allow_html=True)

fitur_info = [
    {"name": "age", "label": "Usia Pasien (Tahun)", "type": "number", "min": 0, "max": 120, "default": 35},
    {"name": "sex", "label": "Jenis Kelamin", "type": "select"},
    {"name": "bmi", "label": "Indeks Massa Tubuh (BMI)", "type": "float", "min": 10.0, "max": 60.0, "default": 24.5},
    {"name": "smoker", "label": "Apakah Pasien Merokok?", "type": "select"},
    {"name": "alcohol_freq", "label": "Konsumsi Alkohol", "type": "select"},
    {"name": "systolic_bp", "label": "Tekanan Darah Sistolik (Atas)", "type": "number", "min": 50, "max": 250, "default": 120},
    {"name": "diastolic_bp", "label": "Tekanan Darah Diastolik (Bawah)", "type": "number", "min": 30, "max": 150, "default": 80},
    {"name": "ldl", "label": "Kadar Kolesterol LDL", "type": "number", "min": 10, "max": 300, "default": 110},
    {"name": "hba1c", "label": "Tingkat HbA1c (Gula Darah)", "type": "float", "min": 3.0, "max": 15.0, "default": 5.4},
    {"name": "visits_last_year", "label": "Jumlah Kunjungan Medis Tahun Lalu", "type": "number", "min": 0, "max": 50, "default": 2},
    {"name": "hospitalizations_last_3yrs", "label": "Jumlah Rawat Inap (3 Tahun Terakhir)", "type": "number", "min": 0, "max": 10, "default": 0},
    {"name": "days_hospitalized_last_3yrs", "label": "Total Hari Dirawat di RS (3 Tahun Terakhir)", "type": "number", "min": 0, "max": 365, "default": 0},
    {"name": "medication_count", "label": "Jumlah Konsumsi Obat Rutin", "type": "number", "min": 0, "max": 30, "default": 1},
    {"name": "proc_imaging_count", "label": "Jumlah Prosedur Rontgen/MRI/Scan", "type": "number", "min": 0, "max": 20, "default": 1},
    {"name": "proc_surgery_count", "label": "Jumlah Tindakan Operasi/Pembedahan", "type": "number", "min": 0, "max": 10, "default": 0},
    {"name": "proc_physio_count", "label": "Jumlah Sesi Fisioterapi", "type": "number", "min": 0, "max": 30, "default": 0},
    {"name": "proc_consult_count", "label": "Jumlah Konsultasi Dokter Spesialis", "type": "number", "min": 0, "max": 30, "default": 2},
    {"name": "proc_lab_count", "label": "Jumlah Tes Laboratorium", "type": "number", "min": 0, "max": 50, "default": 1}
]
total_langkah = len(fitur_info)

# --- 5. LOGIKA FORM INPUT STEP-BY-STEP ---
if st.session_state.inputan <= total_langkah:
    index_saat_ini = st.session_state.inputan - 1
    fitur_sekarang = fitur_info[index_saat_ini]
    nama_fitur = fitur_sekarang["name"]
    
    st.write(f"**Langkah ke-{st.session_state.inputan} Dari {total_langkah}**")
    
    if fitur_sekarang["type"] == "select":
        opsi_pilihan = categories_dict[nama_fitur]
        st.selectbox(fitur_sekarang["label"], options=opsi_pilihan, key=nama_fitur)
    elif fitur_sekarang["type"] == "number":
        st.number_input(fitur_sekarang["label"], min_value=fitur_sekarang["min"], max_value=fitur_sekarang["max"], value=fitur_sekarang["default"], step=1, key=nama_fitur)
    elif fitur_sekarang["type"] == "float":
        st.number_input(fitur_sekarang["label"], min_value=fitur_sekarang["min"], max_value=fitur_sekarang["max"], value=fitur_sekarang["default"], step=0.1, key=nama_fitur)
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("⬅️ Undo", on_click=undo, disabled=(st.session_state.inputan == 1))
    with col2:
        st.button("➡️ Next", on_click=lanjut_step, args=(nama_fitur,))
    with col3:
        st.button("🔄 Reset", on_click=reset)

# --- 6. PROSES PREDIKSI & KOMPARASI ---
elif st.session_state.inputan > total_langkah:
    st.success("🎉 Semua data telah berhasil diisi!")
    
    if st.button("🔍 Jalankan Komparasi Semua Model"):
        st.write("Sedang memproses data dan melakukan scaling...")
        
        data_user = st.session_state.jawaban_user.copy()
        df_input = pd.DataFrame([data_user])
        df_input = df_input[feature_cols]
        
        try:
            for col in ['sex', 'smoker', 'alcohol_freq']:
                le = encoders_dict[col]
                df_input[col] = le.transform(df_input[col].astype(str))
                
            data_scaled = scaler_obj.transform(df_input)
            
            hasil_komparasi_angka = {}
            hasil_komparasi_teks = {}
            
            for nama_model, model_aktif in pilihan_model_dict.items():
                pred = model_aktif.predict(data_scaled)
                nilai_prediksi = int(pred[0])
                
                hasil_komparasi_angka[nama_model] = nilai_prediksi
                hasil_komparasi_teks[nama_model] = "High Risk 🚨" if nilai_prediksi == 1 else "Low Risk ✅"
            
            # TAMPILKAN HASIL UTAMA
            st.write("---")
            st.subheader("Hasil Analisis Risiko Kesehatan Pasien")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("XGBoost (Best)", hasil_komparasi_teks["XGBoost"])
            col2.metric("Logistic Regression", hasil_komparasi_teks["Logistic Regression"])
            col3.metric("Random Forest", hasil_komparasi_teks["Random Forest"])
            col4.metric("Decision Tree", hasil_komparasi_teks["Decision Tree"])
            
            # --- TAMPILKAN TABEL & GRAFIK SEJAJAR (GRID DASHBOARD) ---
            st.write("---")
            kolom_kiri, kolom_kanan = st.columns(2)
            
            df_hasil = pd.DataFrame(list(hasil_komparasi_angka.items()), columns=['Nama Model', 'Nilai Risiko (0=Rendah, 1=Tinggi)'])
            
            with kolom_kiri:
                st.write("### 📋 Tabel Perbandingan")
                st.dataframe(df_hasil, hide_index=True, use_container_width=True)
                
            with kolom_kanan:
                st.write("### 📊 Grafik Batang Risiko")
                df_chart = df_hasil.set_index('Nama Model')
                st.bar_chart(df_chart, use_container_width=True)
                
        except Exception as e:
            st.error(f"Terjadi kegagalan pemrosesan model: {e}")
            
    st.write("---")
    st.button("Ulangi Dari Awal", on_click=reset)