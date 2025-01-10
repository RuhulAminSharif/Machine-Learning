## Working with CSV files

Now, discussion time for working with ``CSV`` file in details.

### 0. Importing pandas
To import import pandas:
```python
import pandas as pd
```
### 1. Opening a Local CSV File
Load a CSV file from a local directory:
```python
df = pd.read_csv('path/to/file.csv')
```

### 2. Opening a CSV File from a URL
If the CSV is hosted online, we can directly load it using the URL:
```python
import requests
from io import StringIO
import pandas as pd

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}

# Perform the GET request with custom headers
req = requests.get(url, headers=headers)

# Check if the request was successful
if req.status_code == 200:
    data = StringIO(req.text)  # Read the text content as a file-like object
    df = pd.read_csv(data)
    print(df.head())
else:
    print(f"Failed to retrieve data. HTTP Status code: {req.status_code}")
```
### 3. `names` Parameter
If the file doesn’t contain column headers, to read the CSV file we can set ``header=None`` and use ``names`` to specify the column names:
```python
import pandas as pd

# Custom column names
column_names = ['Country', 'Population', 'GDP']

# Load the CSV without header and assign custom column names
df = pd.read_csv('file.csv', header=None, names=column_names)
print(df.head())
```
Even if the CSV has headers, we can override them with names:
```python
df = pd.read_csv('file.csv', names=['A', 'B', 'C'], header=0)
```
Here, ``header=0`` tells pandas to skip the first row as a header and use the ``names`` list instead.  

**Notes**  
- The number of names in names should match the number of columns in the data (or the subset specified by usecols).
- If names is specified without ``header=None``, pandas still loads the original header row as data in the first row.

### 4. `sep` Parameter (Delimiter)
To specify a custom delimiter, such as a semicolon (`;`), use the `sep` parameter:
```python
df = pd.read_csv('file.csv', sep=';')
```
This is especially useful for non-standard delimited files like `.tsv` files.

**Some more delimiter:**
- Tab-Separated Files: ```df = pd.read_csv('file.tsv', sep='\t')```
- Space-Separate Files: ```df = pd.read_csv('file.txt', sep=' ')```
- Multiple Character Delimiters(like ::) : ```df = pd.read_csv('file.txt', sep='::', engine='python')```
  - **Note**: When using multi-character separators, we must set engine='python', as the default c engine only supports single-character delimiters.
- Whitespace as Delimiter: ```df = pd.read_csv('file.txt', sep='\s+', engine='python')```

### 5. `index_col` Parameter (Index Column)  
The ``index_col`` parameter in ``pandas.read_csv()`` specifies which column(s) should be used as the index of the resulting DataFrame. By default, pandas creates a numeric index starting at 0, but using ``index_col`` can be helpful when the data already contains a column that uniquely identifies each row, such as an ID column.
```python
df = pd.read_csv('file.csv', index_col='ColumnName')
```

### 6. `header` Parameter (Row to Use as Header)
Select a specific row as the header:
```python
df = pd.read_csv('file.csv', header=1)  # Second row as header (0-based index)
```
If dataset does not have a header row, we can use `header=None` to prevent pandas from treating the first row as headers.  
By default, pandas use ``header=0`` i.e., first row as the header.

### 7. `usecols` Parameter (Loading Specific Columns)
To load only the columns we need:
```python
df = pd.read_csv('file.csv', usecols=['Col1', 'Col2'])
```

### 8. `squeeze` Parameter (Single Column as Series)
If loading a single column, use `squeeze=True` to load it as a Series instead of a DataFrame:
```python
series = pd.read_csv('file.csv', usecols=['SingleColumn'], squeeze=True)
```

### 9. `skiprows` Parameter (Skipping Rows)
Skip specific rows at the start of the file, such as for files with metadata at the top:
```python
df = pd.read_csv('file.csv', skiprows=4)
```

### 10. `nrows` Parameter (Number of Rows to Read)
Read only a specific number of rows:
```python
df = pd.read_csv('file.csv', nrows=500)
```
This is useful when previewing a large dataset.

### 11. `encoding` Parameter (Character Encoding)
Specify the character encoding if the file contains non-ASCII characters:
```python
df = pd.read_csv('file.csv', encoding='ISO-8859-1')
```
If we encounter errors related to special characters, try `encoding='utf-8'`.

### 12. `error_bad_lines` Parameter (Skip Bad Lines)
Skip lines that cannot be parsed, often due to inconsistent columns:
```python
df = pd.read_csv('file.csv', error_bad_lines=False)
```
This parameter can help load data from poorly structured files. However, we should use it carefully, as it may result in loss of data.

### 13. `dtype` Parameter (Setting Data Types)
Assign specific data types to columns:
```python
df = pd.read_csv('file.csv', dtype={'col1': 'int64', 'col2': 'float64'})
```
This is useful to ensure correct data types, prevent memory overuse, and avoid data type conversion issues later.

### 14. Handling Dates (`parse_dates` and `infer_datetime_format`)
Automatically parse columns as dates:
```python
df = pd.read_csv('file.csv', parse_dates=['DateColumn'])
```
With `infer_datetime_format=True`, pandas tries to infer the date format, improving performance.
```python
df = pd.read_csv('file.csv', parse_dates=['DateColumn'], infer_datetime_format=True)
```

### 15. `converters` Parameter (Custom Data Conversion)
Apply custom functions to transform values in specific columns:
```python
df = pd.read_csv('file.csv', converters={'ColumnName': lambda x: x.upper()})
```
This is useful for pre-processing data without additional post-processing steps.

### 16. `na_values` Parameter (Custom Missing Values)
Specify custom representations of missing values:
```python
df = pd.read_csv('file.csv', na_values=['N/A', 'missing'])
```
By default, pandas treats blank fields as NaN, but this can be extended to custom indicators.

### 17. Loading Large Datasets in Chunks (`chunksize`)
For very large files, use `chunksize` to read the file in parts:
```python
chunk_iter = pd.read_csv('file.csv', chunksize=10000)
for chunk in chunk_iter:
    # Process each chunk
    process(chunk)
```
This is memory-efficient and useful for handling files that are too large to fit in memory at once.

### 18. `low_memory` Parameter (Efficient Loading)
If we’re loading a large file with mixed types in columns, avoid dtype guessing with `low_memory=False`:
```python
df = pd.read_csv('file.csv', low_memory=False)
```
This can prevent type inference errors and allow consistent data types in columns.

### 19. `comment` Parameter (Skip Comments)
Skip lines starting with a specified character, such as `#`:
```python
df = pd.read_csv('file.csv', comment='#')
```
Useful for CSV files with comment lines.

### 20. `skipfooter` Parameter (Skipping Footer Rows)
If the file has a footer, skip rows from the end:
```python
df = pd.read_csv('file.csv', skipfooter=2, engine='python')
```
This is especially useful when working with CSVs that include summary data at the end.

### 21. `thousands` Parameter (Handling Thousand Separators)
Specify the thousands separator for numerical values:
```python
df = pd.read_csv('file.csv', thousands=',')
```
This helps correctly load numbers with separators, such as "1,000,000".

### 22. `memory_map` Parameter (Using Memory Mapping for Faster Loading)
Enable memory mapping to speed up reading of large files:
```python
df = pd.read_csv('file.csv', memory_map=True)
```
This is particularly useful on systems with limited memory.
