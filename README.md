## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2024-05-04 084550](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/91bd9d70-31d9-40b7-8622-71a498f7a4c1)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-05-04 084558](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/4ed80133-e88b-4646-aff6-65bc76f2bb57)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-05-04 084609](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/a5a900e9-6a2c-4428-8806-76a1294d06c5)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-05-04 084618](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/09bac396-8a22-411f-9d89-4f7cd2ed18a2)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-05-04 084627](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/296b1f40-89fa-499e-bde2-aee74de51ea8)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-05-04 084635](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/9daf9174-c462-45a4-a0e9-7b6a36bc4b22)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-05-04 084645](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/811ab987-c3c8-4d4b-8495-86c73cf8ca10)

```
pip install --upgrade category_encoders
```
![Screenshot 2024-05-04 084654](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/8fd87639-6307-445e-ba87-ec1955e78e3a)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb=df.copy()
dfb
```
![Screenshot 2024-05-04 084702](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/e0cee882-7a3b-4c37-bf31-284eb9406230)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2024-05-04 084711](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/1db9e9f2-a4f5-406f-940b-316c734d73ae)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-05-04 084721](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/f54f2b59-7f51-4536-b2ce-e7c81f9676ff)

```
df.skew()
```
![Screenshot 2024-05-04 084729](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/efef1316-c96a-4f3c-808f-529d622010b9)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-05-04 084739](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/48e68324-b430-4f98-91e3-b15fb816a1b1)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2024-05-04 084747](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/2376d117-066f-4fc2-b688-68e445393330)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-05-04 084755](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/ca05fcfd-e40f-4404-8de7-e61397da5bb7)

```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-05-04 084805](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/f489e287-0356-45ce-93bb-53cb2fed1f6e)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
```
![Screenshot 2024-05-04 084815](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/44fc9d9e-0c97-45a6-ad83-d34235d13afa)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![Screenshot 2024-05-04 084822](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/513285f7-f17d-4837-96fb-d4081dfa2012)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2024-05-04 084829](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/c9339b16-ea0c-4e2f-8c27-7132621ce7fd)

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![Screenshot 2024-05-04 084838](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/42a8280b-d512-4b62-9b7a-b115c8ba394c)

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![Screenshot 2024-05-04 084847](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/f802cce5-bfe4-497f-b781-d51ec5369236)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![Screenshot 2024-05-04 084858](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/32749625-f24b-4c32-b348-ce751d6a3bd2)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-05-04 084905](https://github.com/Harsayazheni/Expt03-Introduction-to-Data-Science/assets/118708467/34db54f5-b507-4207-8471-654a34a4cb90)

# RESULT:
Hence performing Feature Encoding and Transformation process is Successful.

       
