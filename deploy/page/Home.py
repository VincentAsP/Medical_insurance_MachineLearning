import streamlit as st
import joblib
import pandas as pd  # Pastikan import pandas untuk konversi DataFrame

if "inputan" not in st.session_state:
    st.session_state.inputan = 1
    
if "jawaban_user" not in st.session_state:
    st.session_state.jawaban_user = {}

# Fungsi navigasi
def next(nama_fitur):
    st.session_state.jawaban_user[nama_fitur] = st.session_state[nama_fitur]
    st.session_state.inputan += 1

def undo():
    if st.session_state.inputan > 1:
        st.session_state.inputan -= 1

def reset():
    st.session_state.clear()
    st.session_state.inputan = 1
    st.session_state.jawaban_user = {}

# --- CACHE MODEL ---
# Menggunakan cache agar model tidak di-load ulang setiap kali user mengetik/klik tombol
@st.cache_resource
def load_all_models():
    return {
        "XGBoost": joblib.load('deploy/XGB_pkl'),
        "Linear Regression": joblib.load('deploy/LR_pkl'),
        "Random Forest": joblib.load('deploy/RF_pkl'),
        "Decision Tree": joblib.load('deploy/DT_pkl')
    }

pilihan_model_dict = load_all_models()

# 2. Bagian Header UI
st.title("Medical Insurance 🏥")
st.write("""
**Kelompok:**
1. Aretha Tiurma S.M - 2802468220
2. Muhammad Edgar T.S - 2802545352
3. Vincentius Axel S.P - 2802528554
""")

st.markdown("<h2 style='color:cyan'>Prediksi Health Insurance</h2>", unsafe_allow_html=True)
st.write("Kami akan menganalisis prediksi apakah harga untuk asuransi kesehatan akan tinggi atau tidak. Model yang akan kami gunakan adalah **XGBoost** karena sejauh ini, dengan cara mengukur akurasi, presisi, dan F1-Score dan membandingkan antara Decision Tree, XGBoost, dan Random Forest, kami memilih model XGBoost karena ini adalah model terbaik untuk dataset ini dengan F1-score yang menyentuh angka 0.86, presisi di angka 0.91, dan akurasi di angka 0.90.")

st.markdown("<h2 style='color:cyan'>Penerapan Model</h2>", unsafe_allow_html=True)

fitur = [
    'Nama', 'age', 'sex', 'region', 'urban_rural', 'income', 
    'education', 'marital_status', 'employment_status', 'household_size', 
    'dependents', 'bmi', 'smoker', 'alcohol_freq', 'visits_last_year', 
    'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs', 
    'medication_count', 'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c', 
    'plan_type', 'network_tier', 'deductible', 'copay', 'policy_term_years', 
    'policy_changes_last_2yrs', 'provider_quality', 'risk_score', 
    'annual_medical_cost', 'annual_premium', 'monthly_premium', 'claims_count', 
    'avg_claim_amount', 'total_claims_paid', 'chronic_count', 'hypertension', 
    'diabetes', 'asthma', 'copd', 'cardiovascular_disease', 'cancer_history', 
    'kidney_disease', 'liver_disease', 'arthritis', 'mental_health', 
    'proc_imaging_count', 'proc_surgery_count', 'proc_physio_count', 
    'proc_consult_count', 'proc_lab_count', 'had_major_procedure'
] 
total_langkah = len(fitur)

# Jika masih dalam proses input
if st.session_state.inputan <= total_langkah:
    index_saat_ini = st.session_state.inputan - 1
    nama_fitur_sekarang = fitur[index_saat_ini]
    
    st.write(f"**Langkah ke-{st.session_state.inputan} Dari {total_langkah}**")
    
    st.text_input(
        f"Masukkan nilai untuk {nama_fitur_sekarang}:", 
        key=nama_fitur_sekarang, 
        on_change=next
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Undo", on_click=undo, disabled=(st.session_state.inputan == 1))
    with col2:
        st.button("🔄 Reset", on_click=reset)

# Jika semua input sudah selesai
elif st.session_state.inputan > total_langkah:
    st.success("🎉 Semua data telah berhasil diisi!")
    
    st.write("### Pilih Model Prediksi")
    nama_model_terpilih = st.selectbox(
        "Pilih model machine learning yang ingin digunakan:",
        ("XGBoost", "Linear Regression", "Random Forest", "Decision Tree")
    )
    
    # Ambil model dari dictionary berdasarkan dropdown
    model_aktif = pilihan_model_dict[nama_model_terpilih]
    
    # Tombol untuk melakukan prediksi
    if st.button(f"🔍 Cek Prediksi Asuransi dengan {nama_model_terpilih}"):
        st.write(f"Menyiapkan data untuk **{nama_model_terpilih}**...")
        
        # 1. Ambil data dictionary
        data_user = st.session_state.jawaban_user.copy()
        
        # Hapus fitur 'Nama' karena model ML tidak memproses nama
        if 'Nama' in data_user:
            del data_user['Nama']
            
        # 2. Jadikan Pandas DataFrame
        df_input = pd.DataFrame([data_user])
        
        try:
            # --- ⚠️ PENTING: KONVERSI TIPE DATA ⚠️ ---
            # Karena kamu pakai st.text_input, semua data masuk sebagai String (Teks).
            # Model ML akan error jika dikasih string. Kita harus ubah ke angka.
            
            # Ubah tipe data numerik menjadi integer atau float sesuai kebutuhan
            # (Pastikan kamu menyesuaikan nama kolom di bawah ini jika ada yang kurang/berbeda format)
            kolom_float = ['bmi', 'income', 'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c', 'deductible', 'copay', 'risk_score', 'annual_medical_cost', 'annual_premium', 'monthly_premium', 'avg_claim_amount', 'total_claims_paid']
            
            for col in df_input.columns:
                if col in kolom_float:
                    df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
                else:
                    # Asumsikan sisanya adalah integer atau kolom kategorik yang sudah diinput dalam bentuk angka (0/1/2)
                    # Jika user mengetik huruf ('Male'), to_numeric dengan errors='ignore' akan membiarkannya tetap string,
                    # tapi kamu HARUS melakukan encoding manual jika modelmu butuh format angka.
                    df_input[col] = pd.to_numeric(df_input[col], errors='ignore')
            
            # 3. Lakukan Prediksi
            hasil_prediksi = model_aktif.predict(df_input)
            
            # 4. Ambil nilai hasilnya (biasanya berbentuk array [1] atau [15000])
            nilai_hasil = hasil_prediksi[0] if hasattr(hasil_prediksi, '__len__') else hasil_prediksi
            
            # 5. Tampilkan Output
            st.success(f"### Output Prediksi: {nilai_hasil}")
            
        except Exception as e:
            st.error(f"**Terjadi error saat prediksi:** {e}")
            st.info("""
            **Saran Perbaikan:** Error di atas biasanya terjadi karena *tipe data salah* atau *kolom kategorik belum di-encode*. 
            Pastikan jika modelmu meminta `sex` berformat `0/1`, maka user juga harus menginput angka `0` atau `1` (bukan mengetik 'Male' atau 'Female').
            """)
    
    st.write("---")
    st.button("Ulangi dari Awal", on_click=reset)