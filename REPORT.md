# 醫療資料隱私作業報告：Composition / Intersection Attack

## 組員

| 姓名 | 學號 |
| --- | --- |
| TODO | TODO |
| TODO | TODO |

> 本次分析使用 `data/synthetic_profiled/` 中的隨機合成資料，不含真實病患資料。資料結構仍保留 FAERS、VAERS、MIMIC-III Demo 的欄位形狀，因此可用來示範 composition / intersection attack 的流程。

## 1. 資料集與敏感資訊

| 資料集 | 使用時間範圍 | 使用的主要檔案 | 敏感資訊欄位 |
| --- | --- | --- | --- |
| FAERS | 2004-2012 | `DEMO*.csv`, `INDI*.csv`, `REAC*.csv`, `DRUG*.csv` | indication (`INDI_PT`)、reaction (`PT`)、drug (`DRUGNAME`) |
| VAERS | 2004-2012 | `*VAERSDATA.csv`, `*VAERSSYMPTOMS.csv`, `*VAERSVAX.csv` | symptoms (`SYMPTOM1`-`SYMPTOM5`)、history、current illness、allergies、symptom text |
| MIMIC-III Demo | demo sample | `PATIENTS.csv`, `ADMISSIONS.csv`, `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, `MICROBIOLOGYEVENTS.csv` | diagnosis、ICD9、prescription、microbiology / antibiotic interpretation |

## 2. 可用 QID 欄位

第一性原則下，一個欄位要能放進 intersection key，必須同時滿足：

1. 它描述的是同一個人的同一種屬性。
2. 至少兩個資料來源都有語意相同或可標準化的欄位。
3. 粒度可對齊，例如年齡都轉為 years、日期都轉為 year / quarter。
4. 攻擊者可能從外部知道該欄位，例如性別、年齡、事件年份。
5. 不是資料集內部流水號，例如 `PRIMARYID`, `VAERS_ID`, `subject_id`。
6. 不是本次要保護或推斷的主要敏感值，例如疾病、症狀、用藥、ICD9。

依照這個判準，可以用的欄位如下：

| 比對範圍 | 使用的 QID | 說明 |
| --- | --- | --- |
| FAERS + VAERS + MIMIC | `sex + age_band + age_year` | 三份資料都可標準化的共同病患屬性。MIMIC 日期有 shift，不把 MIMIC 年份當真實年份。 |
| FAERS + VAERS | `sex + age_band + age_year + report_year + report_quarter` | 兩份不良事件資料都可取得性別、年齡、通報年份與通報季度，因此兩兩比對時全部使用。 |
| FAERS + MIMIC | `sex + age_band + age_year` | MIMIC 日期 shift，且 FAERS/MIMIC 沒有共同地理欄位。 |
| VAERS + MIMIC | `sex + age_band + age_year` | VAERS 的州別在 MIMIC 沒有對應欄位；MIMIC 日期 shift。 |

更完整地說，所有欄位可分成三類：

| 類別 | 欄位 | 是否放入主 QID | 理由 |
| --- | --- | --- | --- |
| 直接可用 | `SEX` / `gender`, `AGE` / `AGE_YRS` / derived age, `report_year`, `report_quarter` | 是 | 可跨資料來源標準化，且不是內部 ID 或敏感值。 |
| 單一資料集內可用 | VAERS `STATE`; MIMIC `ethnicity`, `insurance`, `language`, `religion`, `marital_status`, `admission_location`; FAERS `OCCP_COD`, `MFR_SNDR`, `OCCR_COUNTRY`, `REPORTER_COUNTRY` | 否 | 可能有識別力，但不是三方或兩方共同欄位，不能直接作跨來源 key。 |
| 條件式可用 | death / outcome, exact event date, product / drug / vaccine / manufacturer / lot / route | 本報告不放入主要 QID | 若 threat model 假設攻擊者已知道這些事件資訊，可做更強攻擊；但它們常常本身就是敏感事件、暴露或結果。 |
| 不應使用 | `PRIMARYID`, `CASEID`, `VAERS_ID`, `subject_id`, `hadm_id`, `row_id` | 否 | 這些是資料集內部 ID，跨資料來源沒有共同語意。 |
| 敏感值 | FAERS `INDI_PT`, `PT`, `DRUGNAME`; VAERS `SYMPTOM*`, `HISTORY`, `CUR_ILL`, `ALLERGIES`, `SYMPTOM_TEXT`; MIMIC `diagnosis`, `icd9_code`, `drug`, `org_name`, `interpretation` | 否 | 這些是本作業要計算 `P(qid -> s)` 的 `s`。 |

所以「全部都用」在本報告的意思是：對每個比對範圍，使用所有語意一致、可標準化、非內部 ID、非主要敏感值的 QID。完整欄位清單輸出於 `outputs/qid_field_inventory.csv`。

實作程式：

```bash
python3 scripts/composition_attack.py --data-root data/synthetic_profiled --output-dir outputs --c 0.2 --backend auto
```

程式使用 Polars/Arrow columnar pipeline；`--backend auto` 會在大檔案且 cuDF 可用時使用 cuDF 讀取 CSV。

## 3. Intersection Attack 結果

標準化後共有 45,100 筆來源紀錄：

| Source | Records |
| --- | ---: |
| FAERS | 36,000 |
| VAERS | 9,000 |
| MIMIC | 100 |

使用所有共同可用 QID 後，交集結果如下：

| 比對範圍 | QID groups | 候選組合 / pair 數 |
| --- | ---: | ---: |
| FAERS + VAERS + MIMIC | 67 | 362,783 |
| FAERS + VAERS | 3,212 | 146,257 |
| FAERS + MIMIC | 74 | 11,982 |
| VAERS + MIMIC | 67 | 3,103 |

三方交集候選明細共有 10,832 筆，其中 FAERS 8,402 筆、VAERS 2,342 筆、MIMIC 88 筆。前三個三方交集候選群組如下：

| sex | age_band | age_year | FAERS | VAERS | MIMIC | candidate combinations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F | 45-64 | 45 | 210 | 46 | 2 | 19,320 |
| F | 18-29 | 21 | 110 | 47 | 3 | 15,510 |
| M | 00-17 | 0 | 44 | 336 | 1 | 14,784 |

FAERS + VAERS 使用完整 QID 後，前三個候選群組如下：

| sex | age_band | age_year | year | quarter | FAERS | VAERS | candidate pairs |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| F | 65+ | 120 | 2007 | Q4 | 296 | 20 | 5,920 |
| F | 65+ | 120 | 2007 | Q3 | 257 | 23 | 5,911 |
| F | 65+ | 120 | 2012 | Q2 | 195 | 25 | 4,875 |

完整候選病患資訊輸出於：

- `outputs/intersection_three_source_candidates.csv`
- `outputs/intersection_faers_vaers_candidates.csv`
- `outputs/intersection_faers_mimic_candidates.csv`
- `outputs/intersection_vaers_mimic_candidates.csv`
- `outputs/matched_patient_details.csv`

## 4. P(qid -> s) 計算

令 `qid` 為一組 quasi-identifier，`s` 為敏感資訊，例如疾病、症狀、用藥、ICD9 診斷。機率定義如下：

```text
P(qid -> s) =
  count(records with the same qid and containing sensitive value s)
  / count(records with the same qid)
```

程式會將每筆紀錄的 `sensitive_values` 展開後計算，因此同一筆紀錄若同時有多個症狀、診斷或用藥，會分別計入不同的 `s`。

使用完整 QID 後，因為群組更小，部分兩資料集 QID 群組的推斷機率會變高。所有 scope 中保護前最高 `P(qid -> s)` 為 0.80；三方交集 scope 中最高 `P(qid -> s)` 為 0.3846。

保護前三方交集的高風險例子：

| qid scope | qid | sensitive value | probability |
| --- | --- | --- | ---: |
| three-source | F, 18-29, age 17 | synthetic clinical placeholder | 0.3846 |
| three-source | F, 00-17, age 7 | synthetic clinical placeholder | 0.3380 |
| three-source | M, 00-17, age 0 | synthetic clinical placeholder | 0.3281 |

完整機率表輸出於：

- `outputs/qid_sensitive_probability.csv`

## 5. 保護方法：C-bounding

本作業採用 confidence bounding / C-bounding，設定 `C = 0.2`，要求對任一使用的 QID scope 與任一敏感值 `s`：

```text
P(qid -> s) <= C
```

實作方法：

1. 對每個 QID 群組計算所有敏感值的支持度。
2. 若某個敏感值的 `P(qid -> s) > C`，找出含有該敏感值的紀錄。
3. 以 deterministic sensitive-value suppression 移除最少必要的 `(record, sensitive value)` 關聯，而不是刪除整筆候選紀錄。
4. 重新計算，直到該 scope 內所有 QID 群組都滿足 `P(qid -> s) <= 0.2`。

保護成效如下：

| 比對範圍 | 保護前候選紀錄 | 保護後紀錄 | suppressed sensitive values |
| --- | ---: | ---: | ---: |
| FAERS + VAERS + MIMIC | 10,832 | 10,832 | 153 |
| FAERS + VAERS | 30,350 | 30,350 | 15,196 |
| FAERS + MIMIC | 9,290 | 9,290 | 17 |
| VAERS + MIMIC | 2,430 | 2,430 | 1,014 |

保護後所有使用的 QID scope 中，最大 `P(qid -> s)` 皆降至 0.2。

保護後資料與機率表輸出於：

- `outputs/protected_c_bounded_records.csv`
- `outputs/protected_faers_vaers_c_bounded_records.csv`
- `outputs/protected_faers_mimic_c_bounded_records.csv`
- `outputs/protected_vaers_mimic_c_bounded_records.csv`
- `outputs/protected_c_bounded_probability.csv`

## 6. 限制與說明

1. 本 repo 目前使用的是隨機合成資料，因此找到的是「依 QID 無法排除為同一人」的候選群組，不代表真實病患連結。
2. MIMIC-III Demo 有日期 shift；本分析沒有把 MIMIC 日期年份納入三方或 MIMIC 相關兩方 QID。
3. 單一資料集特有欄位沒有拿來做跨資料集 key；若要做同資料集內的 linking attack，可另外納入這些欄位。
4. 合成資料中有 `SYN_*` 與 placeholder text，這些在流程中仍當作敏感值示範機率計算；若改用真實去識別資料，可直接重跑同一支程式。
