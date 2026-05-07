# 👁️ PyEyeTrack: Webcam-Based Global Eye Tracker & Heatmap Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Bu proje, standart bir web kamerası kullanarak işletim sistemi seviyesinde çalışan, kullanıcının ekranda nereye baktığını takip eden ve oturum sonunda detaylı bir **Isı Haritası (Heatmap) ve Fiksasyon Analizi** sunan bir göz takip (eye-tracking) yazılımıdır. 


## ✨ Özellikler (Features)

* **Global Şeffaf Katman (Overlay):** PyQt5 kullanılarak işletim sisteminin üzerine şeffaf ve tıklamaları geçiren (click-through) bir katman açılır. Arka planda oyun oynarken, kod yazarken veya web'de gezinirken çalışmaya devam eder.
* **Gelişmiş Kalibrasyon:** İnsan gözü hareketlerini öğrenmek için Percentile (Yüzdelik) filtreleme kullanır. Anlık kamera hatalarını veya göz kırpmalarını çöpe atarak pürüzsüz bir kalibrasyon sağlar.
* **Dinamik Yumuşatma (Dynamic Smoothing & Deadzone):** Web kameralarının kronik sorunu olan mikro titremeleri engellemek için Medyan filtreleme ve mesafeye duyarlı hareket kilitleri kullanır.
* **Oturum Sonu Analiz Raporu:** `ESC` tuşuna basıldığında oturumu kapatır ve kullanıcının ekranda geçirdiği süreyi analiz ederek şunları sunar:
  * Gaussian Blur destekli yoğunluk ısı haritası (Heatmap).
  * Fiksasyon (odaklanma) noktaları ve saniye cinsinden süreleri.
  * Ekranın 3x3 ızgaraya bölünmüş bölge analizi (Hangi bölgeye % kaç bakıldı).

## 🚀 Kurulum (Installation)

Projeyi bilgisayarınızda çalıştırmak için Python 3.8 veya üzeri bir sürümün yüklü olması gerekmektedir. (Not: MediaPipe uyumluluğu için Python 3.11 önerilir).

**1. Projeyi klonlayın:**
git clone https://github.com/al1code/Webcam-Eye-Tracker.git

**2. Gerekli kütüphaneleri yükleyin:**
pip install opencv-python numpy mediapipe keyboard scipy PyQt5

## Kullanım (Usage)

**C Tuşu (Kalibrasyon)** : Programa gözlerinizin hareket sınırlarını öğretmek için **C** tuşuna basın.
Ekranda turuncu bir uyarı çıkacaktır. Kafanızı sabit tutarak gözlerinizle ekranın 4 köşesine bakın. 
İşiniz bittiğinde tekrar **C** tuşuna basarak kalibrasyonu tamamlayın.

**ESC Tuşu (Çıkış ve Analiz)** : Göz takibi oturumunu sonlandırmak için **ESC** tuşuna basın. 
Program kapanacak ve saniyeler içinde o oturuma ait tüm hareketlerinizi gösteren tam ekran bir analiz raporu sunacaktır.
Bu rapor aynı zamanda heatmap_kayitlar klasörüne resim olarak kaydedilir.

Nasıl Çalışıyor? (How it Works)
Sistem, Google'ın MediaPipe Face Mesh modelini kullanarak yüzdeki 468 referans noktasını tespit eder. Ancak kafanın hareketini (Head Pose) değil, sadece sağ ve sol göz pınarlarına olan mesafeye göre İrisin (gözbebeği) oransal sapmasını hesaplar.

Elde edilen bu çiğ (raw) oranlar:

1.Göz kırpma (blink) filtrelerinden geçer.

2.Medyan (Median) filtre ile anlık gürültülerden arındırılır.

3.Exponential Smoothing ile yumuşatılır.

4.Ölü bölge (Deadzone) mantığı ile görsel titremeler kilitlenir.

5.Bilgisayar ekranının piksel çözünürlüğüne (Örn: 1920x1080) matematiksel olarak yayılır (mapping).

## ⚠️ Kısıtlamalar (Limitations)
Bu proje profesyonel bir donanım (Kızılötesi Lazer) değil, standart RGB Web kamerası kullanmaktadır. Bu nedenle:

**1.Kalibrasyon yapıldıktan sonra kafanın sabit tutulması gerekmektedir. 
Kafa açısının veya kameraya olan uzaklığın değişmesi, matematiksel oranları bozacaktır.**

**2.Çok düşük ışıklı ortamlarda MediaPipe'ın irisi tespit etme doğruluğu düşebilir**

## 📄 Lisans (License)
Bu proje MIT Lisansı ile lisanslanmıştır. Detaylar için LICENSE dosyasına göz atabilirsiniz. 
**Dilediğiniz gibi kullanabilir, değiştirebilir ve geliştirebilirsiniz.**

YAZARLAR

CANER SAL - Ali KEMAL DİLEK

