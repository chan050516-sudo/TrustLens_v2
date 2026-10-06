这份清单整理了大量马来西亚公共注册数据源，但**存在严重的路径错误、命名混淆和定位偏差**。

逐条核查并给出实际入口与修正方案：

---

### 第一梯队：直接下载结构化文件

#### 1. MCMC 持牌通信/邮政服务商

* **为什么找不到？**
1. 官网菜单已重构：旧版路径中的 `Licensing` 现归并在 **`Legal`（法律监管）** 菜单下。
2. 邮政/快递（Postal & Courier）不属于常规电信法（CMA 1998），在 MCMC 体系中单独归类于 **`PSA Registers`**（*Postal Services Act 2012*）。


* **正确入口与路径：**
* **路径：** 访问 [mcmc.gov.my](https://www.google.com/search?q=https://www.mcmc.gov.my) → 顶部导航点击 **Legal** → 选择 **Registers** → 点击 **PSA Registers**（或在 Search 输入 "Register of Licensees Postal Services"）。
* **直接公开直链：** MCMC 长期维护的静态公开表单通常命名为 `List-of-Courier-Services-Licenses_.pdf` 或位于 `skmmgovmy/media/General/pdf/` 目录中。



#### 2. BNM 持牌金融机构名单

* **状态：** 基本正确，但页面结构有出入。
* **问题：** BNM 官网没有名为 `"COMMERCIAL BANKS"` 的一级独立页面。商业银行、伊斯兰银行、投行全部分组挂在 **`Malaysian Financial Sector`** 下。
* **正确路径：**
* `bnm.gov.my` → **Topics** → **Financial Stability** → **Malaysian Financial Sector** → **List of Licensed Financial Institutions**。
* 页面提供了分栏 HTML 表格（商业银行、伊斯兰银行、投资银行等），可直接提取。



#### 3. Bursa Malaysia 上市公司名录

* **存在重大逻辑错误：**
* **不要抓取 SBL 清单！** `SBL ELIGIBLE SECURITIES` 是**证券借贷（Securities Borrowing & Lending）融券标的名单**，并非全量上市公司名录。许多小市值股、停牌股或特定板块股票不在 SBL 范围内。


* **正确获取方式：**
* **方法 A（官方最全目录）：** 访问 `bursamalaysia.com` → **Market Information** → **Equities** → **List of Companies**。选择板块（Main / ACE / LEAP Market），支持直接导出或分页抓取。
* **方法 B（直接 Excel 下载）：** 访问 Bursa Malaysia 的 `Company Directory` 页面，点击 **"Download Excel"** / **"Export to CSV"** 按钮即可一次性获得完整的 Stock Code、Company Name、Market Board、Sector 分类。



#### 4. NPRA 药剂监管局数据

* **状态：** 完全正确，且数据质量极高。
* **修正补充：** 官方开放平台 `data.gov.my` 已经对 NPRA 做了完整的每日自动化镜像，无需写解析器，直接拉取现成数据：
* **获批药品（包含 MAL 号、配方、持牌公司）：** [data.gov.my/data-catalogue/pharmaceutical_products](https://data.gov.my/data-catalogue/pharmaceutical_products)
* **获批药品进口商（带执照号、地址、电话）：** [data.gov.my/data-catalogue/pharmaceutical_importers](https://data.gov.my/data-catalogue/pharmaceutical_importers)
* 格式直接提供 `.csv` 与 `.parquet` 静态直链。



---

### 第二梯队：需要爬虫或 Apify 工具

#### 5. Bursa Malaysia（KLSE Screener）

* **状态：** 逻辑可行，但**不建议优先用 Apify 付费跑**。
* **分析：**
* KLSE Screener 官方公开接口非常薄，其移动端/网页端有现成的轻量 JSON API，或者直接用开源 Python 库（如 `investing-scrapers` 或分析其 XHR 请求）即可抓取 PE/EPS/Dividend，无需买云端 Actor。



#### 6. Malaysian Bar Council 律师公会

* **状态：** 正确。
* **路径：** [legaldirectory.malaysianbar.org.my](https://legaldirectory.malaysianbar.org.my/)。
* **注意：** 该名录采用分页与姓名/执照反查机制，全量遍历（2.5万+律师）会触发单 IP 请求频次限制，爬取时需设置适当延时与代理轮换。

#### 7. MIA 会计师公会名录

* **存在重大信息偏差：**
* **关于 RM 15 费用：** RM 15 针对的是特定认证请求或详细背景报告。
* **公众查询完全免费：** MIA 官网的 **`Firm & Member Directory`**（公共执业验证系统）是免费公开查询的，只是前端做了验证码防御且没有提供一次性批量“导出”按钮。若需数据，需通过自动化表单遍历请求提取事务所（Firm）与特许会计师（CA(M)）信息。



---

### 第三梯队：PDF 解析或商业数据

#### 8. CIDB 承包商名录

* **存在重大来源错误：**
* **不要找 KKR（工程部）官网找 PDF**。KKR 官网只有零散通告，根本不维护动态名录。


* **真正入口：**
* CIDB 的全量在线管理系统为 **CIMS CIDB**（Construction Industry Management System）。
* 查询入口：[cims.cidb.gov.my → Search Registered Contractor (Local/Foreign)](https://cims.cidb.gov.my/smis/regcontractor/reglocalsearchcontractor.vbhtml)。
* 这里记录了所有 G1 到 G7 等级的有效承包商、注册编号与业务资质（PPK/SPKK/STB），需针对该 ASP/VB.NET 表单编写抓取脚本。



#### 9. MQA 认证高等院校 / 课程名录

* **状态：** 正确。
* **路径：** [mqr.mqa.gov.my](https://mqr.mqa.gov.my)。
* **分析：** 包含 MQR（全认证）和 Provisional Accreditation（临时认证）两张主表，按 HEI（高等教育机构）导出，结构整齐。

#### 10. SSM 公司注册数据

* **状态：** 正确。
* **分析：**
* SSM 官方无论是 `ssm-einfo.my` 还是新版 `MyData SSM` 均需按件付费（每份基本资料约 RM 10 - RM 15，简要状态 RM 3）。
* **针对批量需求：** 确实如你所述，如果需要全量清洗后的百万级公司数据字典，找第三方数据集成商（如 Handshakes、CTOS 或 Experian B2B 批发包）通常比直接找 SSM 采买原始数据库要便宜并带有统一结构标准。