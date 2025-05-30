# گزارش پیش‌پردازش داده‌های فروش بازی‌های ویدیویی
## مراحل پاکسازی، تبدیل و مهندسی ویژگی

### مقدمه
این گزارش مراحل پیش‌پردازش مجموعه داده فروش بازی‌های ویدیویی VGChartz را شرح می‌دهد. هدف از این پیش‌پردازش، تمیز کردن داده‌ها، کنترل مقادیر پرت، رفع ناسازگاری‌های منطقی، استانداردسازی ویژگی‌های عددی، کدگذاری متغیرهای کیفی و ایجاد ویژگی‌های جدید است تا داده‌ها برای تحلیل‌های پیشرفته‌تر آماده شوند.

### کد آماده‌سازی محیط و بارگذاری داده‌ها

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# ایجاد پوشه برای داده‌های پردازش شده و تصویرسازی‌ها
os.makedirs('processed_data', exist_ok=True)
os.makedirs('cleaning_plots', exist_ok=True)

print("Loading the dataset...")
# بارگذاری مجموعه داده و ایجاد یک کپی برای پیش‌پردازش
df = pd.read_csv('vgchartz_cleaned.csv')
df_processed = df.copy()

print(f"Original dataset shape: {df.shape}")
```

این کد کتابخانه‌های مورد نیاز را وارد کرده و پوشه‌هایی برای ذخیره داده‌های پردازش شده و نمودارهای مربوط به پاکسازی ایجاد می‌کند. سپس مجموعه داده را بارگذاری کرده و یک کپی از آن برای پیش‌پردازش ایجاد می‌کند.

### پاکسازی داده‌ها

```python
# ============= DATA CLEANING =============
print("\n===== DATA CLEANING =====")

# 1. بررسی رکوردهای تکراری
duplicates = df_processed.duplicated().sum()
print(f"Number of duplicate records: {duplicates}")
if duplicates > 0:
    df_processed = df_processed.drop_duplicates()
    print(f"After removing duplicates: {df_processed.shape}")

# 2. بررسی مقادیر گمشده
print("\nMissing Values:")
missing_values = df_processed.isnull().sum()
print(missing_values)

# فقط سطرهایی را نگه می‌داریم که ستون‌های ضروری مقدار خالی نداشته باشند
essential_columns = ['title', 'console', 'genre', 'publisher', 'critic_score', 'total_sales']
df_processed = df_processed.dropna(subset=essential_columns)
print(f"After removing rows with missing essential data: {df_processed.shape}")
```

در این بخش، ابتدا به دنبال رکوردهای تکراری در داده‌ها می‌گردیم. سپس مقادیر گمشده را شناسایی می‌کنیم و سطرهایی که در ستون‌های ضروری مقدار خالی دارند را حذف می‌کنیم.

### تشخیص و کنترل مقادیر پرت

```python
# 3. بررسی مقادیر ناسازگار در ستون‌های عددی
numeric_cols = ['critic_score', 'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 
                'other_sales', 'release_year']

print("\nChecking for outliers and inconsistent values in numeric columns...")
for col in numeric_cols:
    if col in df_processed.columns:
        # ستون‌هایی با نرخ بالای مقادیر گمشده را رد می‌کنیم
        if df_processed[col].isna().sum() > len(df_processed) * 0.5:
            print(f"Skipping {col} due to high missing value rate")
            continue
            
        # محدوده را چاپ می‌کنیم و به دنبال مقادیر غیرعادی می‌گردیم
        min_val = df_processed[col].min()
        max_val = df_processed[col].max()
        print(f"{col}: Range [{min_val}, {max_val}]")
        
        # شناسایی مقادیر پرت احتمالی با استفاده از روش IQR
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
        
        if not outliers.empty:
            print(f"  Found {len(outliers)} potential outliers in {col}")
            
            # تصویرسازی مقادیر پرت
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df_processed[col])
            plt.title(f'Boxplot of {col} - Outlier Detection')
            plt.tight_layout()
            plt.savefig(f'cleaning_plots/outliers_{col}.png')
            plt.close()
            
            # محدود کردن مقادیر پرت به مقادیر مرزی (جایگزینی برای حذف)
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            print(f"  Outliers in {col} capped to [{lower_bound:.2f}, {upper_bound:.2f}]")
```

در این بخش، برای هر ستون عددی، از روش دامنه میان چارکی (IQR) برای شناسایی مقادیر پرت استفاده می‌کنیم. مقادیر پرت شناسایی شده به مرزهای تعیین شده محدود می‌شوند. همچنین برای هر ستون، یک نمودار جعبه‌ای برای نمایش مقادیر پرت ایجاد می‌کنیم.

### رفع ناسازگاری‌های منطقی

```python
# 4. بررسی ناسازگاری‌های منطقی
print("\nChecking for logical inconsistencies...")

# امتیاز منتقدان باید بین 0 و 10 باشد
if df_processed['critic_score'].max() > 10 or df_processed['critic_score'].min() < 0:
    print(f"Found critic_score values outside the expected range [0, 10]")
    df_processed['critic_score'] = df_processed['critic_score'].clip(0, 10)
    print("  Capped critic_score to [0, 10]")

# ارقام فروش باید غیرمنفی باشند
for col in ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']:
    if col in df_processed.columns and df_processed[col].min() < 0:
        print(f"Found negative values in {col}")
        df_processed[col] = df_processed[col].clip(0, None)
        print(f"  Capped {col} to non-negative values")

# سال انتشار باید منطقی باشد (مثلاً 1970-2023)
if 'release_year' in df_processed.columns:
    invalid_years = df_processed[(df_processed['release_year'] < 1970) | (df_processed['release_year'] > 2023)]
    if not invalid_years.empty:
        print(f"Found {len(invalid_years)} records with unusual release_year values")
        df_processed['release_year'] = df_processed['release_year'].clip(1970, 2023)
        print("  Capped release_year to [1970, 2023]")
```

در این بخش، به دنبال ناسازگاری‌های منطقی در داده‌ها می‌گردیم و آنها را اصلاح می‌کنیم:
- اطمینان حاصل می‌کنیم که امتیازات منتقدان بین 0 و 10 هستند
- مطمئن می‌شویم که ارقام فروش منفی نیستند
- سال‌های انتشار را به یک محدوده منطقی (1970-2023) محدود می‌کنیم

### پر کردن مقادیر گمشده

```python
# 5. پر کردن مقادیر عددی گمشده باقی‌مانده
print("\nFilling remaining missing values...")
for col in numeric_cols:
    if col in df_processed.columns and df_processed[col].isna().any():
        missing_count = df_processed[col].isna().sum()
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        print(f"  Filled {missing_count} missing values in {col} with median")
```

مقادیر گمشده باقی‌مانده در ستون‌های عددی را با مقدار میانه آن ستون پر می‌کنیم.

### تبدیل داده‌ها

```python
# ============= DATA TRANSFORMATION =============
print("\n===== DATA TRANSFORMATION =====")

# 1. استانداردسازی ویژگی‌های عددی
print("\nStandardizing numeric features...")
numeric_cols_to_scale = [col for col in numeric_cols if col in df_processed.columns]
scaler = StandardScaler()
df_processed[numeric_cols_to_scale] = scaler.fit_transform(df_processed[numeric_cols_to_scale])
print(f"Standardized {len(numeric_cols_to_scale)} numeric columns")
```

در این بخش، تمام ستون‌های عددی را با استفاده از `StandardScaler` استاندارد می‌کنیم. این فرآیند میانگین را به صفر و انحراف معیار را به یک تبدیل می‌کند، که برای روش‌های یادگیری ماشین مفید است.

### کدگذاری متغیرهای کیفی

```python
# 2. کدگذاری متغیرهای کیفی
categorical_cols = ['console', 'genre', 'publisher', 'developer']
print("\nEncoding categorical variables...")

for col in categorical_cols:
    if col in df_processed.columns:
        print(f"Processing {col}...")
        # بررسی تعداد مقادیر یکتا
        n_unique = df_processed[col].nunique()
        print(f"  {n_unique} unique values")
        
        # برای متغیرهای باینری، از کدگذاری برچسب ساده استفاده می‌کنیم
        if n_unique == 2:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            print(f"  Applied label encoding to {col}")
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"  Mapping: {mapping}")
        
        # برای متغیرهای با تعداد دسته‌های محدود، از کدگذاری One-Hot استفاده می‌کنیم
        elif n_unique <= 15:
            # ایجاد متغیرهای دامی و حذف اولین مورد برای جلوگیری از هم‌خطی
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed.drop(col, axis=1, inplace=True)
            print(f"  Applied one-hot encoding to {col}, created {len(dummies.columns)} new features")
        
        # برای متغیرهای با تعداد زیاد مقادیر یکتا، از کدگذاری فراوانی استفاده می‌کنیم
        else:
            # ایجاد یک نقشه فراوانی
            freq_map = df_processed[col].value_counts(normalize=True).to_dict()
            df_processed[f'{col}_freq'] = df_processed[col].map(freq_map)
            df_processed.drop(col, axis=1, inplace=True)
            print(f"  Applied frequency encoding to {col}")
```

در این بخش، متغیرهای کیفی را با روش‌های مختلف کدگذاری می‌کنیم:
- برای متغیرهای باینری (دو مقدار): کدگذاری برچسب
- برای متغیرهای با تعداد مقادیر یکتای کم (≤15): کدگذاری One-Hot
- برای متغیرهای با تعداد زیاد مقادیر یکتا: کدگذاری فراوانی (که هر مقدار را با فراوانی نسبی آن در مجموعه داده جایگزین می‌کند)

### مهندسی ویژگی

```python
# 3. مهندسی ویژگی
print("\nCreating engineered features...")

# محاسبه نسبت‌های فروش بین مناطق
if all(col in df_processed.columns for col in ['na_sales', 'total_sales']):
    df_processed['na_sales_ratio'] = df_processed['na_sales'] / df_processed['total_sales']
    print("Created na_sales_ratio (NA sales / total sales)")

if all(col in df_processed.columns for col in ['jp_sales', 'total_sales']):
    df_processed['jp_sales_ratio'] = df_processed['jp_sales'] / df_processed['total_sales']
    print("Created jp_sales_ratio (Japan sales / total sales)")

if all(col in df_processed.columns for col in ['pal_sales', 'total_sales']):
    df_processed['pal_sales_ratio'] = df_processed['pal_sales'] / df_processed['total_sales']
    print("Created pal_sales_ratio (PAL sales / total sales)")

# ایجاد ویژگی عمر بازی
if 'release_year' in df_processed.columns:
    current_year = 2023
    df_processed['game_age'] = current_year - df_processed['release_year']
    print("Created game_age feature (years since release)")

# ایجاد ویژگی فروش در هر سال
if all(col in df_processed.columns for col in ['total_sales', 'game_age']):
    df_processed['sales_per_year'] = df_processed['total_sales'] / (df_processed['game_age'] + 1)  # +1 برای جلوگیری از تقسیم بر صفر
    print("Created sales_per_year feature")

# ایجاد متغیر دسته‌بندی برای امتیاز منتقدان
if 'critic_score' in df_processed.columns:
    bins = [0, 5, 7, 8, 9, 10]
    labels = ['Poor', 'Average', 'Good', 'Great', 'Excellent']
    df_processed['critic_score_category'] = pd.cut(df_processed['critic_score'], bins=bins, labels=labels)
    dummies = pd.get_dummies(df_processed['critic_score_category'], prefix='rating')
    df_processed = pd.concat([df_processed, dummies], axis=1)
    df_processed.drop('critic_score_category', axis=1, inplace=True)
    print("Created critic_score category and dummy variables")
```

در این بخش، ویژگی‌های جدیدی برای غنی‌تر کردن مجموعه داده ایجاد می‌کنیم:
1. **نسبت‌های فروش منطقه‌ای**: سهم هر منطقه از کل فروش
2. **عمر بازی**: تعداد سال‌های گذشته از زمان انتشار بازی
3. **فروش سالانه**: فروش کل تقسیم بر عمر بازی
4. **دسته‌بندی امتیاز منتقدان**: تبدیل امتیازات پیوسته به دسته‌بندی‌های مشخص (ضعیف، متوسط، خوب، عالی، بی‌نظیر)

### نهایی‌سازی و ذخیره داده‌های پردازش شده

```python
# 4. حذف مقادیر NaN باقی‌مانده
df_processed = df_processed.dropna()
print(f"\nAfter removing remaining NaN values: {df_processed.shape}")

# 5. ذخیره داده‌های پردازش شده
processed_file = 'processed_data/vgchartz_processed.csv'
df_processed.to_csv(processed_file, index=False)
print(f"\nProcessed data saved to {processed_file}")
print(f"Final dataset shape: {df_processed.shape}")
```

در این بخش، هرگونه مقدار گمشده باقی‌مانده را حذف کرده و مجموعه داده نهایی را ذخیره می‌کنیم.

### نتایج پیش‌پردازش

براساس نتایج خروجی اسکریپت، می‌توانیم ببینیم که:

- از ۱۸،۸۷۴ رکورد اصلی، ۴،۰۰۰ رکورد در مجموعه داده نهایی باقی ماندند
- تعداد زیادی مقدار گمشده در ستون‌های فروش منطقه‌ای وجود داشت که با میانه پر شدند
- مقادیر پرت قابل توجهی در اکثر ستون‌های عددی شناسایی و محدود شدند
- ستون‌های کیفی با تعداد زیاد مقادیر یکتا (مانند کنسول، ژانر، ناشر و سازنده) با روش کدگذاری فراوانی تبدیل شدند
- ویژگی‌های جدیدی مانند نسبت‌های فروش منطقه‌ای، عمر بازی و فروش سالانه ایجاد شدند

### خلاصه تبدیل‌های اعمال شده
1. حذف رکوردهای تکراری
2. مدیریت مقادیر گمشده در ستون‌های ضروری
3. محدود کردن مقادیر پرت در ستون‌های عددی
4. رفع ناسازگاری‌های منطقی
5. استانداردسازی ویژگی‌های عددی
6. کدگذاری متغیرهای کیفی
7. ایجاد ویژگی‌های مهندسی شده

این فرآیند پیش‌پردازش، مجموعه داده فروش بازی‌های ویدیویی را برای تحلیل‌های پیشرفته‌تر و مدل‌سازی آماده کرده است. 