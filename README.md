# PyEyeTrack — Webcam Eye Tracker & Heatmap Analyzer

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Standart bir web kamerası kullanarak işletim sistemi seviyesinde çalışan, kullanıcının ekranda nereye baktığını takip eden ve oturum sonunda detaylı **Isı Haritası (Heatmap) + Fiksasyon Analizi** sunan açık kaynaklı bir göz takip yazılımıdır.

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| **Şeffaf Overlay** | PyQt5 ile işletim sisteminin üzerine tıklamaları geçiren (click-through) katman |
| **9-Noktalı Kalibrasyon** | Görsel hedeflerle kılavuzlanan, daha doğru kalibrasyon |
| **Göz Kırpma Tespiti** | Eye Aspect Ratio (EAR) ile gerçek zamanlı blink sayımı |
| **Gaze Trail** | Son N bakış noktasını soluklaşan iz olarak gösterir |
| **1080p Kamera Desteği** | Otomatik en yüksek çözünürlük tespiti (1920×1080 → 1280×720 fallback) |
| **Live HUD** | FPS, oturum süresi, göz kırpma sayısı, veri noktası sayısı |
| **Turbo Heatmap** | Perceptually superior TURBO colormap (JET'ten daha iyi) |
| **CSV + JSON Export** | Her oturum sonunda ham veri CSV ve özet JSON kaydı |
| **Çapraz Platform Screenshot** | Windows (Pillow), macOS (screencapture), Linux (scrot) |
| **Model Hash Doğrulama** | İndirilen MediaPipe modeline SHA-256 bütünlük kontrolü |
| **MCP Entegrasyonu** | `@antv/mcp-server-chart` ile Claude Code'dan interaktif grafik üretimi |

---

## Kurulum

Python **3.10** veya üzeri gereklidir.

```bash
git clone https://github.com/al1code/Webcam-Eye-Tracker.git
cd Webcam-Eye-Tracker
pip install -r requirements.txt
```

> **Not:** `keyboard` kütüphanesi Linux'ta root yetkisi gerektirebilir.

---

## Kullanım

```bash
python eye_tracker.py
```

İlk çalıştırmada MediaPipe modeli (~30 MB) otomatik indirilir.

Başlangıç problemlerini hızlı kontrol etmek için:

```bash
py -3.12 eye_tracker.py --self-test
```

Bulunan ve çözülen çalışma zamanı sorunları için [docs/resolved-issues.md](docs/resolved-issues.md) dosyasına bakın.

### Tuş Komutları

| Tuş | Eylem |
|---|---|
| `C` | 9-noktalı kalibrasyon başlat / bitir |
| `T` | Gaze Trail göster / gizle |
| `S` | Anlık heatmap snapshot kaydet |
| `ESC` | Oturumu bitir ve analiz raporunu göster |

### Kalibrasyon İpucu

Kalibrasyon modunda ekranda 9 hedef noktası sırayla belirir. Her noktaya **~1.5 saniye** bakın (kafanızı değil, sadece gözlerinizi hareket ettirin). İşlem bitince `C` tuşuna basmanıza gerek yok — otomatik kapanır.

---

## Çıktılar

`heatmap_kayitlar/` klasörüne şunlar kaydedilir:

- `oturum_YYYYMMDD_HHMMSS.png` — Tam analizli heatmap görüntüsü
- `snapshot_YYYYMMDD_HHMMSS.png` — Anlık snapshot
- `gaze_data_YYYYMMDD_HHMMSS.csv` — Ham gaze koordinatları (timestamp, x, y)
- `session_summary_YYYYMMDD_HHMMSS.json` — Oturum özeti (süre, kırpma sayısı, BPM)

---

## MCP — Claude Code Entegrasyonu

Proje kökündeki `.mcp.json` dosyası, Claude Code'a **@antv/mcp-server-chart** sunucusunu otomatik olarak tanıtır. Bu sayede oturum sonrası CSV verilerinden Claude aracılığıyla interaktif grafikler üretilebilir.

```bash
# Gereksinim: Node.js
npx @antv/mcp-server-chart
```

Örnek Claude Code komutu:
> "heatmap_kayitlar/gaze_data_*.csv dosyasından çizgi grafiği oluştur"

---

## Nasıl Çalışır?

1. **MediaPipe Face Landmarker** — 478 yüz noktasından iris merkezi tespiti
2. **EAR (Eye Aspect Ratio)** — Göz kırpma tespiti
3. **Medyan Filtre** — Anlık gürültü temizleme
4. **Exponential Smoothing** — Pürüzsüz hareket
5. **Deadzone Kilidi** — Mikro titremeleri engeller
6. **Percentile Kalibrasyon** — Gözün gerçek hareket aralığını öğrenir

---

## Kısıtlamalar

- Standart RGB kamera kullanır, kızılötesi donanım değil — kafanın sabit tutulması önerilir.
- Düşük ışıkta MediaPipe iris doğruluğu düşebilir.
- Kalibrasyon sonrası kamera konumunun değişmemesi gerekir.

---

## Güvenlik Notları

- Projede hiçbir API anahtarı, şifre veya kimlik bilgisi bulunmamaktadır.
- MediaPipe modeli resmi Google Storage URL'inden indirilir.
- `MODEL_SHA256` sabiti ile isteğe bağlı SHA-256 bütünlük doğrulaması etkinleştirilebilir.

---

## Lisans

MIT License — dilediğiniz gibi kullanabilir, değiştirebilir ve geliştirebilirsiniz.

**Yazarlar:** CANER SAL · Ali KEMAL DİLEK
