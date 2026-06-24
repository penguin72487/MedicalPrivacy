1.資料集說明

 請使用下列三個公開醫療資料集，

FAERS: 美國FDA所釋放出的藥物不良事件通報資料，從2004開始至今。建議使用老師實驗室建置的系統iADRs中已經過資料清洗的資料(2004-2019)。

VAERS (https://vaers.hhs.gov/data.html): 美國FDA/CDC所釋放出的疫苗施打後出現的不良反應通報事件資料，從1990開始至今

MIMIC-III Demo (https://physionet.org/content/mimiciii-demo/1.4/): MIMIC-III資料集(https://physionet.org/content/mimiciii/1.4/)的抽樣本，原MIMIC-III資料為美國Beth Israel Deaconess Medical Center重症加護病房(2001-2012)病患的住院照護紀錄，含有4萬多個病人，MIMIC-III Demo是從中隨機抽取100個病人的資料。注意: MIMIC-III 資料集的時間或某些欄位有作Shift處理，請詳細看其說明。

2. 作業內容

請模擬多份資料來源情境，進行Intersection (Composition) attack，設法找出同時出現在這三份資料中的病人紀錄。若找不出，至少嘗試同時出現在其中兩份資料集(例如FAERS和VAERS)的病人資料。

(1) 說明如何進行交叉比對(用那些QID欄位，如sex, age)，並列出這些病患的詳細個人資訊與敏感資訊(如疾病)。

(2) 從這些資料計算當你得知每個病人的QID，可以推斷出其具有何種敏感隱私(如疾病或過敏)的機率，即

P(qid -> s)，詳細請參考上課投影片。

(3) 請設計可以保護此種攻擊的方法與程式。例如達成C-bounding模型要求(請參考上課投影片Topic 4-2, p. 72-74)

3. 其他補充說明:

(1) 請以分組(2-3人)的方式完成此作業

(2) 請選三份資料有重疊的年份，即2004-2012

(3) 請將程式碼、找出的病患資料紀錄(.csv檔)，報告說明(md檔)壓縮成一個檔案，由其中一人上傳。報告內容務必說明組員名字與學號

4. 使用真實資料執行方式

本 repo 目前可以用 `data/synthetic_profiled/` 的隨機合成資料示範流程；若要改跑真實資料，請把真實資料放在另一個資料夾，例如 `data/real/`，避免和合成資料混在一起。

重要提醒：

- 不要把真實原始資料 commit 到 git。
- `outputs_real/` 會包含從真實資料推得的候選病患紀錄與敏感資訊，也應視為敏感輸出。
- MIMIC-III Demo 請依 PhysioNet 使用規範下載與使用。

建議資料夾結構如下：

```text
data/real
├── 2004Q1_2012Q4
│   ├── 2004Q1
│   │   └── 2004Q1
│   │       ├── DEMO04Q1.csv
│   │       ├── DRUG04Q1.csv
│   │       ├── INDI04Q1.csv
│   │       ├── REAC04Q1.csv
│   │       └── ...
│   ├── 2004Q2
│   └── ...
├── 2004-2012VAERSData
│   ├── 2004VAERSData
│   │   ├── 2004VAERSDATA.csv
│   │   ├── 2004VAERSSYMPTOMS.csv
│   │   └── 2004VAERSVAX.csv
│   ├── 2005VAERSData
│   └── ...
└── mimic-iii-clinical-database-demo-1.4
    ├── PATIENTS.csv
    ├── ADMISSIONS.csv
    ├── DIAGNOSES_ICD.csv
    ├── D_ICD_DIAGNOSES.csv
    ├── PRESCRIPTIONS.csv
    ├── MICROBIOLOGYEVENTS.csv
    └── ...
```

FAERS 的季度資料夾若不是雙層，例如 `2004Q1/DEMO04Q1.csv`，也可以跑；程式會遞迴尋找 `DEMO*.csv`。MIMIC 檔案若是 `.csv.gz`，請先解壓成 `.csv`。

使用本機環境：

```bash
mamba activate fintech
```

若要在新環境安裝套件，至少需要：

```bash
mamba install -c conda-forge polars pyarrow
# 若有 NVIDIA GPU 且已設定 RAPIDS channel，可再安裝 cuDF。
# fintech 環境已包含 cudf 時可略過。
```

如果 FAERS 下載的是 ASCII `.TXT`，先轉成 CSV：

```bash
python3 scripts/convert_faers_txt_to_csv.py --root data/real/2004Q1_2012Q4
```

檢查資料結構與欄位：

```bash
python3 scripts/inspect_dataset_structure.py \
  --root data \
  --output-dir dataset_reports_real
```

執行 intersection / composition attack 與 C-bounding：

```bash
python3 scripts/composition_attack.py \
  --data-root data/ \
  --output-dir outputs_real \
  --c 0.2 \
  --backend auto
```

如果你的真實資料直接放在 `data/` 底下，而不是 `data/real/`，則改用：

```bash
python3 scripts/composition_attack.py \
  --data-root data \
  --output-dir outputs_real \
  --c 0.2 \
  --backend auto
```

執行時預設會顯示 tqdm 進度條；若要在背景執行或輸出乾淨 log，可加上 `--no-progress`。

`--backend auto` 會以 Polars/Arrow 為主，並在偵測到 cuDF 且 CSV 大於門檻時用 GPU 讀取大檔。若要強制使用 GPU CSV reader，可用：

```bash
python3 scripts/composition_attack.py \
  --data-root data \
  --output-dir outputs_real \
  --c 0.2 \
  --backend cudf \
  --cudf-min-mb 0
```

若資料很多，建議先用 `--backend auto`；小檔案強制 cuDF 反而可能因 GPU 啟動成本較慢。

主要輸出檔：

| 檔案 | 說明 |
| --- | --- |
| `outputs_real/source_records.csv` | 標準化後的三資料來源紀錄 |
| `outputs_real/qid_field_inventory.csv` | 所有可用 / 不適合使用的 QID 欄位說明 |
| `outputs_real/intersection_three_source_candidates.csv` | 同時出現在 FAERS、VAERS、MIMIC 的 QID 候選群組 |
| `outputs_real/intersection_faers_vaers_candidates.csv` | FAERS + VAERS 的完整 QID 候選群組 |
| `outputs_real/intersection_faers_mimic_candidates.csv` | FAERS + MIMIC 的候選群組 |
| `outputs_real/intersection_vaers_mimic_candidates.csv` | VAERS + MIMIC 的候選群組 |
| `outputs_real/matched_patient_details.csv` | 三方交集候選病患詳細紀錄與敏感資訊 |
| `outputs_real/qid_sensitive_probability.csv` | `P(qid -> s)` 機率表 |
| `outputs_real/protected_c_bounded_probability.csv` | C-bounding 保護後的機率表 |
| `outputs_real/summary.json` | 本次執行的總結數字 |

跑完真實資料後，請用 `outputs_real/summary.json` 的數字更新 `REPORT.md` 中的交集數、候選紀錄數、`P(qid -> s)` 與 C-bounding 結果。

最後打包交作業：

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

zip_path = Path("medical_privacy_homework_real.zip")
if zip_path.exists():
    zip_path.unlink()

paths = [Path("REPORT.md"), Path("scripts/composition_attack.py")]
paths += sorted(Path("outputs_real").glob("*"))

with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
    for path in paths:
        zf.write(path, path.as_posix())

print(zip_path)
PY
```
