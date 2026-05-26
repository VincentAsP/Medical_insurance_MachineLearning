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
    
    # Ubah nama fungsinya
def lanjut_step(nama_fitur):
    st.session_state.jawaban_user[nama_fitur] = st.session_state[nama_fitur]
    st.session_state.inputan += 1

# --- CACHE MODEL ---
# Menggunakan cache agar model tidak di-load ulang setiap kali user mengetik/klik tombol
@st.cache_resource
def load_all_models():
    return {
        "XGBoost": joblib.load('deploy/XGB.pkl'),
        "Linear Regression": joblib.load('deploy/LR.pkl'),
        "Random Forest": joblib.load('deploy/RF.pkl'),
        "Decision Tree": joblib.load('deploy/DT.pkl')
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
        on_change=lanjut_step,args=(nama_fitur_sekarang,)
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Undo", on_click=undo, disabled=(st.session_state.inputan == 1))
    with col2:
        st.button("🔄 Reset", on_click=reset)

# Jika semua input sudah selesai
# Jika semua input sudah selesai
elif st.session_state.inputan > total_langkah:
    st.success("🎉 Semua data telah berhasil diisi!")
    
    st.write("### 📊 Komparasi Model Prediksi")
    st.write("Klik tombol di bawah ini untuk melihat perbandingan hasil prediksi dari keempat model secara bersamaan.")
    
    # Tombol untuk melakukan prediksi menggunakan semua model
    if st.button("🔍 Jalankan Semua Model"):
        st.write("Menyiapkan data dan menganalisis...")
        
        # 1. Ambil data dictionary
        data_user = st.session_state.jawaban_user.copy()
        
        # Hapus fitur 'Nama'
        if 'Nama' in data_user:
            del data_user['Nama']
            
        # 2. Jadikan Pandas DataFrame
        df_input = pd.DataFrame([data_user])
        
        try:
            # 3. Konversi Tipe Data
            kolom_float = ['bmi', 'income', 'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c', 'deductible', 'copay', 'risk_score', 'annual_medical_cost', 'annual_premium', 'monthly_premium', 'avg_claim_amount', 'total_claims_paid']
            
            for col in df_input.columns:
                if col in kolom_float:
                    df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
                else:
                    df_input[col] = pd.to_numeric(df_input[col], errors='ignore')
            
            # 4. Lakukan Prediksi untuk SEMUA model
            hasil_komparasi = {}
            
            # Looping untuk menjalankan prediksi pada tiap model di dictionary
            for nama_model, model_aktif in pilihan_model_dict.items():
                hasil_prediksi = model_aktif.predict(df_input)
                # Ambil nilai hasilnya dan pastikan formatnya float/angka
                nilai_hasil = float(hasil_prediksi[0] if hasattr(hasil_prediksi, '__len__') else hasil_prediksi)
                hasil_komparasi[nama_model] = nilai_hasil
            
            # 5. Tampilkan Output UI Komparasi
            st.write("---")
            st.subheader("Hasil Analisis")
            
            # Tampilkan nilai di bagian atas menggunakan UI kolom Streamlit
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("XGBoost", f"{hasil_komparasi['XGBoost']:.2f}")
            col2.metric("Linear Regression", f"{hasil_komparasi['Linear Regression']:.2f}")
            col3.metric("Random Forest", f"{hasil_komparasi['Random Forest']:.2f}")
            col4.metric("Decision Tree", f"{hasil_komparasi['Decision Tree']:.2f}")
            
            st.write("---")
            
            # Tampilkan dalam bentuk Tabel dan Grafik secara berdampingan
            tabel_col, grafik_col = st.columns([1, 2])
            
            # Ubah dictionary hasil menjadi DataFrame agar mudah dibuat grafik
            df_hasil = pd.DataFrame(list(hasil_komparasi.items()), columns=['Nama Model', 'Nilai Prediksi'])
            
            with tabel_col:
                st.write("**Tabel Data:**")
                st.dataframe(df_hasil, hide_index=True)
                
            with grafik_col:
                st.write("**Grafik Perbandingan:**")
                # Set index ke Nama Model agar sumbu X pada chart sesuai
                df_chart = df_hasil.set_index('Nama Model')
                st.bar_chart(df_chart)
                
        except Exception as e:
            st.error(f"**Terjadi error saat prediksi:** {e}")
            st.info("""
            **Saran Perbaikan:** Error di atas biasanya terjadi karena *tipe data salah* atau *kolom kategorik belum di-encode*. 
            Pastikan format input sesuai dengan data latih (training data) sebelumnya.
            """)
    
    st.write("---")
    st.button("Ulangi dari Awal", on_click=reset)