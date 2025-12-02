# package功能
## warren：爬蟲、資料清理
## vitruvius：資料分析
## StevenTricks：常用功能
### 使用邏輯：執行warren先建立資料庫，並且清理資料，再用vitruvius去進行運算，產生可用的結果;透過StevenTricks從旁輔助。
***
>warren：
>> + 步驟
> >1. 每天執行crawler確保資料庫最新
> >2. crawler盡量不要和cleaned同時執行，log會有不同步的情形
> >3. crawler執行完，執行cleaned
> >4. 完成
> >+ 功能
> >1. 目前支援證交所13項資料爬蟲自動下載和清理
> >2. 儲存格式以sqlite3為主
