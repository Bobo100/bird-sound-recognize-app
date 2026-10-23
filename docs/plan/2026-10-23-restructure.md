# 2026-10-23 專案整理與特徵擷取修正

## 背景

這是 2022 碩論的 Android app，把 [BirdSoundRecognize](https://github.com/Bobo100/BirdSoundRecognize) 訓練出的 ResNet 放到手機上即時辨識。
README 的核心宣稱是「app 端用 JLibrosa 重現訓練時的特徵擷取」。這次整理實際驗證了這件事，發現**並沒有完全一致**，因此除了整理結構，也修正了特徵擷取。

## 怎麼驗證 app 和訓練端的特徵

這個環境不能編譯 Android（`dl.google.com` 被網路政策擋住，裝不了 Android SDK / AGP），但 JLibrosa 是純 Java：
1. 把 `JLibrosa/` 加上一個 `android.os.Build` stub，用 `javac` 單獨編譯。
2. 寫一個 harness，照 `RealTimeRecognize3.startRecord()` 的順序跑：16-bit PCM → pre-emphasis → `generateMelSpectroGram` → `amplitudeToDb` → 上下翻轉 → 取 128×256 → `normalize`。
3. 同一段 1 秒音訊（研究 repo `data/audio/` 的錄音，量化成 16-bit），在 Python 跑訓練端的 `utils/audio.melspec`，照 `cv2.imwrite` 的方式轉成 0–255。
4. 逐像素比較。

## 發現的問題

### 1. 80 dB 截斷的基準錯誤（`AudioFeatureExtraction.amplitudeToDb`）
librosa 的 `top_db` 是 `max(log_spec) - 80`，app 寫成 `log_spec[0][0] - 80`（左上角那一格）。
左上角通常比峰值低很多，所以 app 幾乎不截斷，低能量區域被拉得比訓練時大。

### 2. `amin` 下限不同
librosa `amplitude_to_db` 預設 `amin=1e-5`，而且是在 [-1, 1] 的浮點訊號上算的；app 用 `1e-10`，訊號又是 16-bit 整數刻度。
安靜的片段（例如錄音開頭），訓練端實際的下限是 -55 dB 而不是 -80 dB，app 和訓練端就不一樣。
mel 頻譜（power=1）和訊號刻度是線性關係，所以把 `amin` 設為 `1e-5 × 32768` 即可與 librosa 完全等價，呼叫端不用改。

### 3. `normalize()` 的初始值錯誤（`RealTimeRecognize3.kt`）
`max_float = 0F`、`min_float = 255F`。dB 值都 ≤ 0，所以最大值永遠停在 0。
librosa 的峰值常常落在被裁掉的第 257 個 frame，這時訓練端的最大值是負的（例如 -1.5 dB），app 卻用 0，整張圖偏暗。
另外 app 用 `toInt()` 無條件捨去，訓練端 PNG 是四捨五入。

### 4. UI 在背景執行緒更新
辨識結果在錄音的 `Thread` 裡直接 `setText`，Android 只允許在 UI 執行緒操作 View，可能閃退。

### 5. 從來沒有請求麥克風權限
`startRecord()` 只檢查權限，沒授權就直接 return，使用者按 Record 沒反應。

### 6. 版控雜物
`.gradle/`（建置快取）、`.idea/`（109 個 library 設定檔等）、`local.properties`（含本機 SDK 路徑 `C:\Users\Bobo\...`）都被 commit。

### 7. 範本殘留
`res/navigation/nav_graph.xml` 指向不存在的 `FirstFragment` / `SecondFragment`；相機相關 drawable、十幾個字串沒有任何地方引用。

## 驗證結果（每格 0–255，平均絕對誤差）

| 片段 | 原本 | 修正後 |
|---|---|---|
| Acanthis flammea 第 0 秒（安靜） | 96.4 | 0.00 |
| Acanthis flammea 第 3 秒 | 26.0 | 0.00 |
| Accipiter gentilis 第 0 秒 | 12.5 | 0.00 |
| Accipiter gentilis 第 3 秒 | 1.0 | 0.00 |
| Abroscopus albogularis 第 0 / 1 / 4 秒 | — | 0.00 |

修正後最大誤差 1（浮點誤差），相關係數 1.0000。
改過的 `normalize()` 另外抽出來用 kotlinc 2.0.21 編譯並測試：輸出範圍 0–255、全靜音時回傳 0 不會閃退（`roundToInt()` 遇到 NaN 會丟例外，所以加了 `maxMin > 0` 判斷）。

> 備註：比對時發現研究 repo 改用新版 librosa 後，`pad_mode` 預設從 `reflect` 變成 `constant`，會和 2022 年的訓練資料不一致；JLibrosa 是 `reflect`。這個修正放在研究 repo 的 PR。

### 對已部署模型的影響
`assets/resnet_model.tflite` 是用正確的（librosa）特徵訓練的，所以修正後 app 餵進模型的資料才和訓練時一樣，辨識結果應該會變好，但**沒有在實機上量過準確率**。

## 做了什麼
- `JLibrosa/process/AudioFeatureExtraction.java`：修正 #1、#2。
- `RealTime2/RealTimeRecognize3.kt`：修正 #3、#4、#5。
- `.gitignore`；`.gradle/`、`.idea/`、`local.properties` 移出版控（本機檔案不受影響）。
- 刪除 #7 的未使用資源。
- README 改寫成和實際行為一致（特徵擷取對照表、結構、建置方式）。

## 刻意保留 / 這次沒做
- **相依套件清理（等能編譯再做）**：`build.gradle` 裡有很多沒用到的套件，`ffmpeg-kit-full`（APK 會多幾十 MB）、`tensorflow-android:1.13.1`、`guava`、`tensorflow-lite-task-audio`、`play-services-*`、`navigation-*`、`exifinterface`、`libs/TarsosDSP-nightly.jar`，`tensorflow-lite` 也重複宣告兩次。
  但 `org.apache.commons.math3`（JLibrosa 會用到）目前是透過 `JTransforms` 間接引入的，拿掉之前要先改成直接宣告。沒辦法編譯就改相依套件風險太高，所以留到可以 build 時處理。
- AGP 7.2 / Kotlin 1.5 / `jcenter()` 升級：同上。
- 套件名稱（`RealTime2`、`Variable` 沒有放在 `com.*` 底下）、`applicationId com.example.mysoundtest`：改了會影響已安裝的 app 和整個專案的 import，效益不大。

## 合併前請在 Android Studio 確認
1. Build 成功。
2. 第一次開啟會跳出麥克風權限。
3. Record → 結果持續更新、不閃退；stop 正常停止。
