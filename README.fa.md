# تحلیل گراف پروفایل‌ها

این پروژه یک نمونهٔ آفلاین برای استخراج ساختارمند پروفایل و تحلیل رابطه‌های سازمانی است. داده‌های پیش‌فرض کاملاً **synthetic** هستند: اشخاص، شرکت‌ها، دانشگاه‌ها و فعالیت‌ها ساختگی‌اند. در توسعه هیچ درخواست واقعی به LinkedIn ارسال نشده است. صفحه‌های HTML در `data/fixtures/` با همان `parser` مورد استفادهٔ adapter اختیاری پردازش می‌شوند.

## اجرا

پیش‌نیاز: Python 3.11+؛ نصب وابستگی‌ها با `pip` انجام می‌شود. از ریشهٔ مخزن:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make run
make test
```

`make install` نیز محیط مجازی را با `python3 -m venv` می‌سازد و با `pip` نصب می‌کند. اگر `make` در دسترس نبود، `PYTHONPATH=src python -m lgi.cli run --source fixture` را اجرا کنید.

`make run` بدون اینترنت کار می‌کند و منبع آن `fixture` است. برای تولید نمونهٔ بزرگ‌تر می‌توان از `.venv/bin/python` همراه `PYTHONPATH=src` و گزینهٔ `--n 50` یا `--n 500` استفاده کرد؛ اجرای تحلیل روی نمونه‌های بزرگ‌تر زمان بیشتری می‌برد. adapterهای `export` و `manual` فقط ورودی‌ای را می‌خوانند که کاربر خودش فراهم کند. adapter `live` خاموش است و فقط با `--source live --i-understand-tos` و فایل URL کاربر فعال می‌شود؛ ممکن است `robots.txt`، صفحهٔ ورود یا شرایط استفاده مانع درخواست شوند.

## خروجی‌ها

- `data/profiles.json` و `data/profiles.csv/`: دادهٔ استخراج‌شده و جدول‌های تخت
- `output/graph.json` و `output/graph.png`: گراف لایه‌ای و تصویر آن
- `output/report.md`: گزارش تولیدشده از اجرای واقعی کد، شامل coverage، centrality، clustering و validation
- `output/tables/`: CSV جدول‌های گزارش
- `output/`: نمودارهای similarity، ranking، dendrogram و stability

نتایج پیش‌فرض تنها رفتار pipeline روی دادهٔ synthetic را نشان می‌دهند. تعداد کم پروفایل‌ها برای نتیجه‌گیری آماری یا پیش‌بینی دربارهٔ افراد واقعی کافی نیست. جزئیات طراحی و محدودیت‌ها در `notes.md` و راهنمای کامل در `README.md` آمده‌اند.
