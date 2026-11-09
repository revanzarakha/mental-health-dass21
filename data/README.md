# Dataset

`dass21_deidentified.csv` berisi 443 respons kuesioner DASS-21 yang digunakan
pada skripsi. Dataset sudah dide-identifikasi sebelum dipublikasikan:

- kolom `Timestamp`, `Umur`, `Jenis Kelamin`, dan `Pekerjaan` dihapus;
- hanya 21 item DASS-21 (skala 0-3) dan tiga skor domain (0-42) yang disertakan;
- tidak ada nama, email, nomor telepon, atau alamat IP.

Data mentah yang masih mengandung metadata responden tidak dipublikasikan.

## Kolom

| Kolom | Keterangan |
|---|---|
| 21 kolom item | Jawaban DASS-21, skala 0 (Tidak Pernah) sampai 3 (Sering Sekali) |
| `Depresi_Score` | Skor domain depresi (0-42) |
| `Kecemasan_Score` | Skor domain kecemasan (0-42) |
| `Stress_Score` | Skor domain stres (0-42) |

Label kategori (Normal, Ringan, Sedang, Berat, Sangat Berat) **tidak** disimpan
di CSV. Label diturunkan ulang oleh `train.py` dari skor domain memakai cut-off
DASS-21 yang dipublikasikan. Ini menjaga satu sumber kebenaran dan menghindari
label yang tidak konsisten.

## Catatan etis

Dataset ini adalah sampel kenyamanan (convenience sample) dari survei, bukan
sampel acak populasi umum. Distribusinya didominasi kategori Berat/Sangat Berat,
sehingga hasil model tidak boleh digeneralisasikan ke populasi luas.

Data ini hanya untuk keperluan pembelajaran dan portofolio. Jangan gunakan untuk
diagnosis atau pengambilan keputusan klinis.
