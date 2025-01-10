## Working with JSON files

### **1. Opening a Local JSON File**
To load a local JSON file into a pandas DataFrame, use `pd.read_json()` with the file path.

```python
import pandas as pd
import json

# Writing a sample JSON file
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Country": ["USA", "Canada", "UK"]
}
with open("data.json", "w") as f:
    json.dump(data, f)

# Reading the JSON file
df = pd.read_json("data.json")
print(df)
```

### **2. Opening a JSON File from a URL**
We can directly load JSON data from an online source using `pd.read_json()` with the URL.

```python
url = "https://api.github.com/repos/pandas-dev/pandas/issues?per_page=5"
df = pd.read_json(url)
df.head()
```

### **3. `orient` Parameter**
The `orient` parameter specifies the structure of the JSON data. It determines how the JSON is interpreted and converted into a DataFrame or Series.

#### Common Options:
- **`split`**: Dictionary with keys `index`, `columns`, and `data`.
- **`records`**: List of dictionaries (each dictionary represents a row).
- **`index`**: Dictionary of dictionaries (keys are row indices).
- **`columns`**: Dictionary of lists (keys are column names).
- **`values`**: A 2D array.

```python
# using split
data = '{"columns":["Name","Age"],"data":[["Alice",25],["Bob",30]]}'
df = pd.read_json(data, orient="split")
df
```

```python
# using records
data = '[{"Name": "Alice", "Age": 25}, {"Name": "Bob", "Age": 30}]'
df = pd.read_json(data, orient="records")
print(df)
```

### **4. `typ` Parameter**
The `typ` parameter specifies whether to return a DataFrame or a Series:
- **`"frame"`**: Returns a DataFrame (default).
- **`"series"`**: Returns a Series.

```python
data = '{"Name": ["Alice", "Bob"], "Age": [25, 30]}'
series = pd.read_json(data, typ="series")
print(series)
```

### **5. `dtype` Parameter**
The `dtype` parameter allows to specify the data type of columns in the resulting DataFrame.

```python
data = '{"Name": ["Alice", "Bob"], "Age": ["25", "30"]}'
df = pd.read_json(data, dtype={"Age": int})
print(df)
```

### **6. Convert Axes**
The `convert_axes` parameter specifies whether to convert the axes of the DataFrame or Series to pandas-compatible types.

### **7. Convert Dates**
If `convert_dates=True`, pandas will attempt to parse date strings into `datetime` objects.

```python
data = '{"Date": ["2024-01-01", "2024-01-02"], "Value": [10, 20]}'
df = pd.read_json(data, convert_dates=True)
print(df)
```

### **8. `keep_default_dates`**
If `False`, pandas will skip automatic date conversion. Use this if dates in the data are not in a standard format or if they should remain as strings.

### **9. `precise_float`**
When working with floating-point numbers, setting `precise_float=True` improves the precision during parsing.

```python
data = '{"Value": [0.12345678912345678, 0.98765432198765432]}'
df = pd.read_json(data, precise_float=True)
print(df)
```
### **10. `date_unit`**
Defines the time unit for parsing dates, e.g., `ms` (milliseconds), `s` (seconds).

```python
data = '{"Date": [1609459200000, 1609545600000]}'
df = pd.read_json(data, convert_dates=True, date_unit="ms")
print(df)
```
### **11. Encoding**
Use the `encoding` parameter to specify the file encoding, e.g., `utf-8`, `latin1`.

### **12. `lines`**
If the JSON file contains line-delimited JSON objects (one object per line), use `lines=True`.

```python
data = '{"Name": "Alice", "Age": 25}\n{"Name": "Bob", "Age": 30}'
df = pd.read_json(StringIO(data), lines=True)
print(df)
```

### **13. `chunksize`**
For large JSON files, load the data in chunks to avoid memory overload.

```python
for chunk in pd.read_json("large_file.json", chunksize=1000):
    print(chunk.head())
```

### **14. Compression**
Supports compressed JSON files (e.g., `gzip`, `bz2`, `zip`, `xz`).

```python
df = pd.read_json("compressed_file.json.gz", compression="gzip")
```

### **15. `nrows`**
Load only the first `n` rows of the file. Useful for quickly inspecting a dataset.

```python
df = pd.read_json("large_file.json", nrows=100)
```

### **16. Engine**
Specifies the parser engine. Commonly used options:
- `"python"`: The Python-based parser (slower, but handles more complex cases).
- `"c"`: The C-based parser (faster).

```python
df = pd.read_json("data.json", engine="c")
```