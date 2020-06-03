# Helpful Functions

### Motivation
Helpful plotting and calculation functions I refer to often.

### Plotting
**Five Thirty Eight Theme**
```
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

x = np.linspace(0, 10)
np.random.seed(19680801)

fig, ax = plt.subplots()

ax.plot(x, np.sin(x) + x + np.random.randn(50))
ax.plot(x, np.sin(x) + 0.5 * x + np.random.randn(50))
ax.plot(x, np.sin(x) + 2 * x + np.random.randn(50))
ax.plot(x, np.sin(x) - 0.5 * x + np.random.randn(50))
ax.plot(x, np.sin(x) - 2 * x + np.random.randn(50))
ax.plot(x, np.sin(x) + np.random.randn(50))
ax.set_title("'fivethirtyeight' style sheet")

plt.show()
```

**FiveThirtyEight Color Palette**
```
fivethirtyeight_colors = ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']
```

**KDE Plot**
```
fig, ax = plt.subplots(figsize=(18, 8))
sns.kdeplot(df['col1'], shade=True, cut=True, label='Column 1')
sns.kdeplot(df['col2'], shade=True, cut=True, label='Column 2')
sns.kdeplot(df['col3'], shade=True, cut=True, label='Column 3')

plt.title('Comparison of Three Distributions')
ax.set(xlabel='Metric', yticklabels=[])
ax.legend().set_title('Column No.')
plt.tight_layout()
plt.show()
```

**Bar Plot w/ Percentages**
```
fig, axs = plt.subplots(figsize=(18, 8))
sns.barplot(x="col1", y="col2", data=df, estimator=lambda x: len(x) / len(df) * 100, color='#30a2da')
plt.title('Barplot of Percentages')
plt.tight_layout()    
plt.show()
```

**Line Plot**
```
fig, ax = plt.subplots(figsize=(18, 8))
sns.lineplot(x='col1', y='col2', data=df, hue='col3')
plt.title('Lineplot', fontsize=22)

ax.set(xlabel='Column 1', ylabel='Column 2')
plt.legend(fontsize='small', title_fontsize='8')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Swarm Plot**
```
fig, ax = plt.subplots(figsize=(10,8))
sns.swarmplot(x='Sex', y='Age', hue='Survived', data=df)
plt.title('Survival by Sex and Age')
plt.show()
```

**Correlation Matrix**
```
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(10,8))
cmap = sns.color_palette('coolwarm')
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
            cbar_kws={'shrink':.5}, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()
```

**Three Axes**
```
fig, axs = plt.subplots(figsize=(20,5), nrows=1, ncols=3)
sns.countplot(df['col1'], ax=axs[0])
axs[0].set(xlabel='Score', ylabel='Count', title='Metric #1')
sns.countplot(df['col2'], ax=axs[1])
axs[1].set(xlabel='Score', ylabel='Count', title='Metric #2')
sns.countplot(df['col3'], ax=axs[2])
axs[2].set(xlabel='Score', ylabel='Count', title='Metric #1')
plt.suptitle('Hypothetical Columns', fontsize=22)
plt.tight_layout()
fig.subplots_adjust(top=0.8)
display(fig)
```

**Three Axes w/ Centered Plot**
```
fig, axs = plt.subplots(figsize=(20,10), nrows=1, ncols=3)
sns.barplot(x="col1", y="col2", data=df, ax=axs[1])
axs[1].set(xlabel='Column1', ylabel='Column2')
axs[0].remove()
axs[2].remove()
plt.title('Column 1 Metric by Column 2 Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Four Axes**
```
fig, axs = plt.subplots(figsize=(18, 8), nrows=2, ncols=2, sharex=True, sharey=True)
sns.kdeplot(df['col1'], shade=True, cut=False, ax=axs[0,0])
axs[0,0].set(title='Plot #1')

sns.kdeplot(df['col2'], shade=True, cut=False, ax=axs[0,1])
axs[0,1].set(title='Plot #2')

sns.kdeplot(df['col3'], shade=True, cut=False, ax=axs[1,0])
axs[1,0].set(title='Plot #3')

sns.kdeplot(df['col4'], shade=True, cut=False, ax=axs[1,1])
axs[1,1].set(title='Plot #4')

plt.suptitle('Four Axes Plot', fontsize=22)
plt.tight_layout()
fig.subplots_adjust(top=0.8)
display(fig)
```

**Plot Duel Axes**
```
fig, ax = plt.subplots()
ax2 = ax.twinx()
sns.barplot(x='col1', y='col2', data=df, color='royalblue', ax=ax)
sns.lineplot(x='col3', y='col4', data=df, color='orange', ax=ax2)
ax.set_ylabel('Y-Label #1', size=12)
ax2.set_ylabel('Y-Label #2', size=12)
ax.set_xlabel('X-Label', size=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=55, fontsize=8)
blue_patch = mpatches.Patch(color='royalblue', label='Booked Productivity')
yellow_patch = mpatches.Patch(color='orange', label='In-Season Units')
plt.legend(handles=[blue_patch, yellow_patch], loc='upper right')
plt.title("Title", size=20)
plt.tight_layout()
display(fig)
```

**Format Tick Marks**
```
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

    return new_tick_format

fig, ax = plt.subplots(figsize=(18, 8))
sns.kdeplot(salary_df.query("Tm == 'DEN'")['Salary'], shade=True, cut=True, label='Denver')
sns.kdeplot(salary_df.query("Tm != 'DEN'")['Salary'], shade=True, cut=True, label='Rest of League')
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
ax.set(xlabel='Salary', yticklabels=[], title='2019 Salary Distribution Denver vs. Rest of League\nData Source: Basketball Reference')
plt.tight_layout()
plt.show()
```

### Calculations
**Gini Index**
```
def gini_calc(arr):
  try:
  # try statement catches all instances in which >0 units were sold within a style in a given time period and returns the gini coefficient
    # Order NET_SALES_UNITS in ascending order
    sorted_arr = sorted(arr)
    height, area = 0, 0
    # Loop through NET_SALES_UNITS
    for value in sorted_arr:
      # Add to cumulative sum
      height += value
      # (Cumulative sum - units sold of specific product color)/2
      area += height - value / 2.
    # fair_area is the area under the line of equality
    fair_area = height * len(arr) / 2
    # output is the area under the lorenz curve divided by the area under the line of equality
    output = (fair_area - area) / fair_area
    return output
  except:
  # except statement catches all instances in which zero units were sold within a style in a given time period and returns None
  # Ex: Free Run sold 0 units in SP2014 and then would throw an error in the gini calculation so we instead impute None
    return None
```

**T-Test**
```
from scipy import stats
print('T-Test')
t2, p2 = stats.ttest_ind(df['col1'], df['col2'])
print("t = " + str(round(t2, 4)))
print("p = " + str(round(p2, 4)))
```

### Pandas Styling
**Tables**
```
import imgkit

df_styled = (df.style
               .set_table_styles(
                 [{'selector': 'tr:nth-of-type(odd)',
                   'props': [('background', '#eee')]},
                  {'selector': 'tbody td', 'props': [('font-family', 'arial')]},
                  {'selector': 'thead', 'props': [('font-family', 'arial')]},
                  {'selector': 'tr:nth-of-type(even)',
                   'props': [('background', 'white')]},
                  {'selector':'th, td', 'props':[('text-align', 'center')]}])
              .set_properties(subset=['col1'], **{'text-align': 'left'})
              .set_caption('Table Title')
              .hide_index())

html = df_styled.render()
imgkit.from_string(html, 'table.png',
                        options={'width': 2925,
                                'disable-smart-width': '',
                                'zoom':3.3})
```
