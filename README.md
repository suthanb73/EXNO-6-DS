# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
```
import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[3,6,2,7,1]
sns.lineplot(x=x,y=y)
```
![image](https://github.com/user-attachments/assets/af5e0060-f920-40f2-9c71-c7e0715da2d3)
```
df = sns.load_dataset("tips")
df
```
![image](https://github.com/user-attachments/assets/f099e49b-a63a-4141-adfa-376efdd2651e)
```
sns.lineplot(x="total_bill",y="tip",data=df,hue="sex",linestyle="solid",legend="auto")
```
![image](https://github.com/user-attachments/assets/2539b790-f752-41fa-991a-1da980aaa703)
```
x=[1,2,3,4,5]
y1=[3,5,2,6,1]
y2=[1,6,4,3,8]
y3=[5,2,7,1,4]
sns.lineplot(x=x,y=y1)
sns.lineplot(x=x,y=y2)
sns.lineplot(x=x,y=y3)
plt.title('Multi-line plot')
plt.xlabel(' x label')
plt.ylabel('Y label')
```
![image](https://github.com/user-attachments/assets/47802e8f-371a-4a08-8c22-4344a42326af)
```
import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.load_dataset('tips')
avg_total_bill=tips.groupby('day')['total_bill'].mean()
avg_tip=tips.groupby('day')['tip'].mean()
plt.figure(figsize=(8,6))
p1=plt.bar(avg_total_bill.index,avg_total_bill,label='Total Bill')
p2=plt.bar(avg_tip.index,avg_tip,bottom=avg_total_bill,label='Tip')
plt.xlabel('Day of the week')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Day')
plt.legend()
```
![image](https://github.com/user-attachments/assets/929f87ec-afbd-47ba-87e9-02bd47b7ae87)
```
avg_total_bill=tips.groupby('time')['total_bill'].mean()
avg_tip=tips.groupby('time')['tip'].mean()
p1=plt.bar(avg_total_bill.index,avg_total_bill,label='Total Bill',width=0.4)
p2=plt.bar(avg_tip.index,avg_tip,bottom=avg_total_bill,label='Tip',width=0.4)
plt.xlabel('Time of Day')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Time of Day')
plt.legend()
```
![image](https://github.com/user-attachments/assets/f6064f9a-d479-4410-be05-88da835271db)
```
import matplotlib.pyplot as plt

years = range(2000, 2011)
apple = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931, 0.934, 0.937, 0.9375, 0.9372, 0.939]
oranges = [0.926, 0.941, 0.930, 0.923, 0.918, 0.907, 0.904, 0.901, 0.898, 0.9, 0.896]

plt.figure(figsize=(12, 6))

# Plot stacked bar chart
plt.bar(years, apple, label='Apple', color='#ff9999')
plt.bar(years, oranges, bottom=apple, label='Oranges', color='#ffcc99')

plt.title("Stacked Bar Chart of Apple and Orange Data (2000-2011)")
plt.xlabel("Year")
plt.ylabel("Values")
plt.xticks(years)
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/2ab29799-72d8-45b3-a2d1-c4b50f7e7d3b)
```
import seaborn as sns
import matplotlib.pyplot as plt
dt = sns.load_dataset('tips')
sns.barplot(x='day', y='total_bill', hue='sex', data=dt, palette='Set1')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill')
plt.title('Total Bill by Day and Gender')
plt.show()
```
![image](https://github.com/user-attachments/assets/86f442d2-8a22-4ce1-b37d-e472010e47c6)
```
import pandas as pd
tit=pd.read_csv("titanic.csv")
tit
```
![image](https://github.com/user-attachments/assets/a869644c-b39d-433c-8fca-907978677c32)
```
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow')
plt.title("Fare of passenger by Embarked Town")
```
![image](https://github.com/user-attachments/assets/b6dd4add-a92f-4901-a291-5c61b959893e)
```
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow',hue='Pclass')
plt.title("Fare of passenegers Embarked Town ,Divided by class")
```
![image](https://github.com/user-attachments/assets/3b1634b5-1cc0-4d04-bf59-229bb0470d35)
```

import seaborn as sns

tips
sns.load_dataset('tips')

sns.scatterplot(x= 'total_bill', y='tip', hue='sex',data=tips)

plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')
```
![image](https://github.com/user-attachments/assets/b9c6192a-d8b9-4130-ae05-042a6e0e14ab)
```
import seaborn as sns
import numpy as np
import pandas as pd
np.random.seed(1)
num_var = np.random.randn(1000)
num_var =pd.Series(num_var,name="Numerical Variable")
num_var
```
![image](https://github.com/user-attachments/assets/ef5fd390-f115-45fd-ab8c-00e32af8a0c8)
```
sns.histplot(data=num_var,kde=True)
```
![image](https://github.com/user-attachments/assets/f7e30fae-9e96-40fd-8f07-636ad72063c7)
```
sns.histplot(data=df,x="Pclass",hue="Survived",kde=True)
```

![image](https://github.com/user-attachments/assets/9ba89573-a1d8-4387-ac21-1dc1c91d7d4a)
```
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)
marks = np.random.normal(loc=70,scale=10,size=100)
marks
```

![image](https://github.com/user-attachments/assets/5ad8414c-4440-4b1c-9bde-2b59f3072d72)
```

sns.histplot(data=marks, bins=18, kde=True, stat='count', cumulative=False, multiple='stack', element='bars', palette='set', shrink=0.7) 
plt.xlabel('Marks')
plt.ylabel('Density')
plt.title('Histogram of students Marks')
```

![image](https://github.com/user-attachments/assets/58a7e32e-d628-47e0-b197-e550f19d0594)
```
import seaborn as sns 
import pandas as pd
tips = sns.load_dataset('tips')
sns.boxplot(x=tips['day'], y=tips['total_bill'], hue=tips['sex'])
```

![image](https://github.com/user-attachments/assets/14062d72-61c3-4296-8212-f0b09d2dc8a9)
```
sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, linewidth=2, width=0.6,boxprops={ "facecolor": "lightblue", "edgecolor": "darkblue"},
whiskerprops={"color": "black", "linestyle":"-", "linewidth": 1.5 }, capprops={ "color": "black", "linestyle":"--", "linewidth": 1.5})
```

![image](https://github.com/user-attachments/assets/12e566f8-d6be-4db8-908f-81c27c2a2ed8)
```
sns.boxplot(x='Pclass',y='Age',data=df,palette='rainbow')
plt.title("Age by Passengers Class,Titanic")
```

![image](https://github.com/user-attachments/assets/c912b836-888a-4edf-bc82-6b8348b40807)
```

sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, linewidth=2, width=0.6,
palette="Set3", inner="quartile")

plt.xlabel("Day of the week")
plt.ylabel("Total Bill")
plt.title("violin Plot of Total Bill by Day and Smoker Status")
```

![image](https://github.com/user-attachments/assets/c2cbdf81-6389-4360-ae63-937882fb24e3)
```
import seaborn as sns
import matplotlib.pyplot as plt

# Load the tips dataset
tips = sns.load_dataset("tips")

# Plot the KDE
sns.kdeplot(data=tips, x='total_bill', hue='time', multiple='fill', linewidth=3, palette='Set2', alpha=0.8)

plt.title("Total Bill Distribution by Time")
plt.show()

```

![image](https://github.com/user-attachments/assets/fb78a488-35db-4211-bad7-c300e8430919)
```
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='layer',linewidth=3,palette='Set2',alpha=0.8)
```

![image](https://github.com/user-attachments/assets/1f33aac5-7733-4fa0-8199-3db36585ace5)
```
import seaborn as sns
import matplotlib.pyplot as plt

# Load the tips dataset
tips = sns.load_dataset("tips")

# Plot the KDE
sns.kdeplot(data=tips, x='total_bill', hue='time', multiple='stack', linewidth=3, palette='Set2', alpha=0.8)

plt.title("Total Bill Distribution by Time")
plt.show()
```

![image](https://github.com/user-attachments/assets/74e74ff1-a2e9-46bb-a81a-f808218fcb35)
```
data=np.random.randint(low = 1,high =100,size=(10,10))
print("data to be plotted:\n")
print(data)
```

![image](https://github.com/user-attachments/assets/81a32246-cb3e-4dea-ba26-e53482b2cfc4)
```
hm=sns.heatmap(data=data,annot=True)
```

![image](https://github.com/user-attachments/assets/dd93a23e-6fa6-4bdc-81f9-906716708a82)

# Result:
Thus the given Data Visualization using Seaborn library is executed successfully.
