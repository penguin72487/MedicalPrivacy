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

(3) 請將程式碼、找出的病患資料紀錄(excel檔)，報告說明(word檔)壓縮成一個檔案，由其中一人上傳。報告內容務必說明組員名字與學號
