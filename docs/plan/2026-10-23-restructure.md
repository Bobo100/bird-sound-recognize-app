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

### 4. pre-emphasis 係數順序相反（code review 時發現）
`FIR` 把 `impulseResponse[0]` 乘在**目前**的樣本上，所以 `floatArrayOf(-0.95F, 1.0F)` 算的是 `y[n] = x[n−1] − 0.95·x[n]`，不是訓練時的 `x[n] − 0.95·x[n−1]`。
另外 FIR 是 Activity 的成員，狀態會延續到下一個窗口和下一次錄音，訓練端則是每段 1 秒各自從 `y[0] = x[0]` 開始。
→ 改成 `floatArrayOf(1.0F, -ALPHA)`，並且每個窗口建立新的 FIR。

### 5. UI 在背景執行緒更新
辨識結果在錄音的 `Thread` 裡直接 `setText`，Android 只允許在 UI 執行緒操作 View，可能閃退。

### 6. 從來沒有請求麥克風權限
`startRecord()` 只檢查權限，沒授權就直接 return，使用者按 Record 沒反應。

### 7. 錄音執行緒的競態條件（code review 時發現，main 原本就有）
按 stop（或 `onStop`）時 UI 執行緒把 `mAudioRecord` 設成 null，但錄音執行緒可能正要執行 `mAudioRecord!!.read()` → NullPointerException 閃退；`isRecording` 也不是 `@Volatile`。快速 stop → Record 時，舊執行緒會繼續讀新的 `AudioRecord`，兩個執行緒同時使用不支援多執行緒的 TFLite interpreter（修正後兩個執行緒不再共用 recorder，而且 stop 時立刻 `stop()` 舊的 recorder 釋放麥克風，舊執行緒處理完當前窗口後再 `release()`）。`read()` 只檢查 `ERROR_INVALID_OPERATION`，其他錯誤碼或讀到較少樣本時仍當作完整資料。

### 8. 版控雜物
`.gradle/`（建置快取）、`.idea/`（109 個 library 設定檔等）、`local.properties`（含本機 SDK 路徑 `C:\Users\Bobo\...`）都被 commit。

### 9. 範本殘留
`res/navigation/nav_graph.xml` 指向不存在的 `FirstFragment` / `SecondFragment`；相機相關 drawable、十幾個字串沒有任何地方引用。

## 驗證結果（每格 0–255，平均絕對誤差）

| 片段 | 原本（main） | 修正後 |
|---|---|---|
| `Acanthis flammea flammea_1.wav` 第 0 秒（安靜） | 96.5 | 0.00 |
| `Acanthis flammea flammea_1.wav` 第 3 秒 | 20.5 | 0.00 |
| `Accipiter gentilis fujiyamae_2.wav` 第 0 秒 | 14.2 | 0.00 |
| `Accipiter gentilis fujiyamae_2.wav` 第 3 秒 | 17.9 | 0.00 |
| `Abroscopus albogularis fulvifacies_3.wav` 第 0 / 1 / 4 秒 | — | 0.00 |

「原本」是 main 的程式碼原封不動（含 FIR 係數錯誤、`normalize` 初始值、無條件捨去），每個窗口重新開始 FIR。第一版文件的這一欄是用「正確的 pre-emphasis + 原本的 dB / normalize」算的，對抗性驗證指出不是真正的原始 app，已更正。

修正後最大誤差 1（浮點誤差），相關係數 1.0000。harness 的 pre-emphasis 是 `FIR.kt` 的逐行移植；用 app 原本的係數 `[-0.95, 1.0]` 時，同樣的片段平均誤差 0.2～17.4。
改過的 `normalize()` 另外抽出來用 kotlinc 2.0.21 編譯並測試：輸出範圍 0–255、全靜音時回傳 0 不會閃退（`roundToInt()` 遇到 NaN 會丟例外，所以加了 `maxMin > 0` 判斷）。

> 備註：比對時發現研究 repo 改用新版 librosa 後，`pad_mode` 預設從 `reflect` 變成 `constant`，會和 2022 年的訓練資料不一致；JLibrosa 是 `reflect`。這個修正放在研究 repo 的 PR。

### 對已部署模型的影響
對抗性驗證用 `resnet_model.tflite` 跑了研究 repo 9 個錄音的 251 段 1 秒音訊，比較「app 特徵」和「訓練端特徵」的 top-1 是否一致：原本 125/251，修正後 251/251。**沒有在實機上量過準確率。**

## 做了什麼
- `JLibrosa/process/AudioFeatureExtraction.java`：修正 #1、#2。
- `RealTime2/RealTimeRecognize3.kt`：修正 #3～#7。錄音迴圈改成：每個錄音執行緒持有自己的 `AudioRecord` 並在結束時釋放；只有 `@Volatile recordingThread` 指向自己時才繼續；依實際讀到的樣本數填窗口；讀取失敗就停止；推論以 `synchronized` 保護。拒絕權限時顯示 Toast。
- `.gitignore`；`.gradle/`、`.idea/`（保留共用的 `codeStyles/`、`dictionaries/`）、`local.properties` 移出版控（本機檔案不受影響）。
- 刪除 #9 的未使用資源。
- `app/build.gradle` 只保留用得到的套件：core-ktx、appcompat、material、constraintlayout、tensorflow-lite(-gpu)，並直接宣告 JLibrosa 需要的 `commons-math3`（原本是透過 JTransforms 間接引入）。移除 navigation、exifinterface、play-services、ffmpeg-kit-full、guava、tensorflow-android 1.13、task-audio、JTransforms、`libs/TarsosDSP-nightly.jar`、重複宣告的 tflite；`junit` 改成 `testImplementation`；移除已停止服務的 `jcenter()`。
- README 改寫成和實際行為一致（特徵擷取對照表、結構、建置方式）。

### 編譯驗證
不能跑 Gradle，但可以用 Maven Central 上的 Robolectric `android-all`（API 31 的完整 Android framework）、TFLite 2.9.0、commons-math3，加上 `AppCompatActivity` / `ActivityCompat` / `R` 的最小 stub，以 javac + kotlinc 2.0.21 編譯 `RealTime2/`、`JLibrosa/` 全部原始碼。修改後可以編譯；另外故意放一個錯誤確認確實會失敗。
這也證明了程式碼只需要上面保留的那幾個套件（資源只用到 material / appcompat 的 theme 和 ConstraintLayout）。

## 刻意保留 / 這次沒做
- AGP 7.2 / Kotlin 1.5 升級：沒有 Android SDK 無法驗證。
- 每個窗口都重新建立 JLibrosa（mel filterbank）、float↔double 陣列複製：原本就有的效能問題，和正確性無關。
- 處理窗口時錄音執行緒沒在讀，所以窗口之間的音訊會被跳過（README 已註明）。改成生產者／消費者需要較大的重構。
- 套件名稱（`RealTime2`、`Variable` 沒有放在 `com.*` 底下）、`applicationId com.example.mysoundtest`：改了會影響已安裝的 app 和整個專案的 import，效益不大。

## Code review 與對抗性驗證

| 來源 | 發現 | 處理 |
|---|---|---|
| review | pre-emphasis 係數順序相反、狀態延續 | 修正（#4） |
| review | stop 時 NPE、stop/start 兩個執行緒、`read()` 錯誤碼 | 修正（#7） |
| review | 權限被拒絕沒有提示、旋轉時重複請求 | Toast + `savedInstanceState == null` 才請求 |
| review | README「每秒」不精確、pre-emphasis 列錯誤 | 修正 README |
| review | dB 峰值第二次掃描多餘 | 簡化成固定 -80 dB 下限 |
| review | `.idea/` 整個忽略會丟掉共用 code style | 保留 `codeStyles/`、`dictionaries/` |
| review | NaN 時 `roundToInt()` 會丟例外 | 不修：輸入是 16-bit PCM，`amplitudeToDb` 輸出一定在 [-80, 0]（對抗性驗證也確認） |
| review | 每個窗口重建 filterbank 等效能問題 | 不修（原本就有） |
| 對抗 | 獨立 harness（264 段，含靜音、削波、正弦、脈衝等合成訊號）證實 FIR 問題；修正後 0/264 段誤差 > 1 | 已修 |
| 對抗 | README「184 種」錯誤，實際 185 種（labels 檔最後沒有換行） | 修正 |
| 對抗 | 若訓練端用 librosa ≥ 0.9 的 `pad_mode='constant'`，即使修正後最大誤差 187 | 研究 repo 明確設定 `SPEC_PAD_MODE='reflect'`（2022 年實際用哪一種無法確定，見研究 repo 的 plan） |
| 對抗（第二輪） | 新 HEAD：264 段全部誤差 ≤ 1、top-1 / top-3 251/251；`RealTime2/` 以真正的 TFLite 2.9.0 jar 加 `-Werror` 編譯通過；移除的套件沒有殘留引用 | 確認 |
| 對抗（第二輪） | 取消權限對話框時 `grantResults` 是空的，會誤顯示「未取得權限」 | 修正 |
| 對抗（第二輪） | `AudioRecord` 初始化失敗（`getMinBufferSize` 錯誤、麥克風被占用）會在 UI 執行緒丟例外 | 檢查 `bufferSize` 和 `state`，失敗時顯示 Toast |
| 對抗（第二輪） | TFLite interpreter 從未關閉（每次旋轉洩漏一個）；辨識中的例外會讓 app 閃退 | 不修：原本就有，關閉 interpreter 要和錄音執行緒同步，記在未來可做 |
| 對抗（第二輪） | plan 的「原本」欄不是真正的原始 app | 用 main 原始碼重算並更正 |
| review（第三輪） | `startRecording()` 可能靜默失敗；建構子可能丟例外；`bufferSize` 只在建立時算一次；stop 後舊 recorder 仍占著麥克風 | 檢查 `recordingState`、try/catch、每次開始時重算、stop 時立即 `stop()`；錯誤訊息移到 `strings.xml` |
| review（第三輪） | 取消權限對話框時沒有任何提示 | 不修：再按 Record 會重新詢問 |

## 合併前請在 Android Studio 確認
1. Build 成功。
2. 第一次開啟會跳出麥克風權限。
3. Record → 結果持續更新、不閃退；stop 正常停止。

## 追加（2026-10-23）：commit 與 PR 格式

新增 `CONTRIBUTING.md` 和 `.github/pull_request_template.md`，三個 repo 內容相同：
- commit 採用 [Conventional Commits](https://www.conventionalcommits.org/)（`type(scope): subject`，英文、祈使句、72 字元以內，body 說明原因）。
- PR 標題用同樣的格式，但最多 65 字元（squash 時 GitHub 會加上 ` (#N)`），內容分成 Summary / Changes / Verification / Test plan。repo 設定裡的 squash 預設訊息要選「Pull request title and description」。
- 這次三個分支上的 commit 訊息也照這個規則整理過（只改訊息、不改內容），因為分支是新建、尚未合併的。
- 工作流程：分支（worktree）→ `docs/plan` → review 與對抗性驗證 → PR → **squash merge**。

**不重寫 main 的舊歷史。** main 上的 commit 都已公開，改寫要 force-push，會讓所有 clone 對不上；而且舊訊息多半是「Update README.md」，重寫也補不回當時的原因。改用 squash merge，讓之後 main 上每個 PR 都是一個格式正確的 commit，分支上的中間 commit 不會留在 main。
