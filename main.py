
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import csv

# 1. Membaca gambar RGB
gambar = iio.imread("jalan.jpg")

# Menghapus channel alpha jika gambar memiliki transparansi
if gambar.ndim == 3 and gambar.shape[2] == 4:
    gambar = gambar[:, :, :3]

# Memastikan gambar memiliki 3 channel RGB
if gambar.ndim != 3 or gambar.shape[2] != 3:
    raise ValueError("Gambar harus berupa gambar berwarna RGB.")

# 2. Mengubah RGB menjadi grayscale
gambar_gray = np.round(
    0.299 * gambar[:, :, 0].astype(float)
    + 0.587 * gambar[:, :, 1].astype(float)
    + 0.114 * gambar[:, :, 2].astype(float)
).astype(np.uint8)

# 3. Menghitung histogram intensitas 0-255
histogram = np.bincount(
    gambar_gray.ravel(), minlength=256
)

# 4. Menampilkan jumlah piksel setiap intensitas
print("Jumlah piksel:", gambar_gray.size)
print("Intensitas | Jumlah piksel")

for intensitas in range(256):
    print(f"{intensitas:10} | {histogram[intensitas]}")

# 5. Menyimpan jumlah piksel ke file CSV
with open("hasil_histogram.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Intensitas", "Jumlah Piksel"])
    for intensitas in range(256):
        writer.writerow([intensitas, histogram[intensitas]])

# 6. Menyimpan gambar grayscale
iio.imwrite("gambar_grayscale.png", gambar_gray)

# 7. Membuat plot histogram
plt.figure(figsize=(10, 5))
plt.bar(range(256), histogram, width=1.0)
plt.title("Histogram Gambar Grayscale")
plt.xlabel("Intensitas Piksel (0-255)")
plt.ylabel("Jumlah Piksel")
plt.xlim(0, 255)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Menyimpan histogram dan menampilkannya
plt.savefig("histogram_grayscale.png", dpi=150)
plt.show()

# 8. Mencari intensitas yang paling dominan
intensitas_dominan = np.argmax(histogram)

print("\nIntensitas paling dominan:", intensitas_dominan)
print("Jumlah piksel dominan:", histogram[intensitas_dominan])
print("Total piksel histogram:", histogram.sum())
