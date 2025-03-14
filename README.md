# Market-Basket-Analysis
## Project Overview
This project analyzes customer purchasing behavior at Supermarket X using association rule mining. The goal is to identify frequently bought items and generate actionable insights for marketing strategies.

## Dataset Description
- The dataset contains historical transaction records from Supermarket X.
- Each row represents a transaction, with items purchased listed as columns.
- Missing values are represented as NaN and are removed during preprocessing.

## Objective
The objective is to apply the Apriori algorithm to identify association rules, which will help supermarket owners create effective marketing plans to boost sales.

## Methodology
1. **Data Preprocessing**
   - Convert transactions into a list format.
   - Encode the transactions using `TransactionEncoder`.
   - Transform the encoded transactions into a DataFrame for analysis.

2. **Frequent Itemset Mining**
   - Apply the Apriori algorithm to identify frequent itemsets with a minimum support threshold of 1%.
   - Extract rules based on the lift metric with a minimum threshold of 1.0.

3. **Key Insights**
   - Most frequently bought items:
     - Almonds (2.03%)
     - Avocado (3.32%)
     - Barbecue Sauce (1.08%)
     - Black Tea (1.43%)
     - Body Spray (1.15%)
   - These items appear in at least 1% of transactions, indicating popularity.

4. **Data Visualization**
   - **Bar Chart**: Displays the top 10 most frequently bought items.
   - **Network Graph**: Illustrates relationships between associated items.
   - **Heatmap**: Highlights the strength of item associations (lift values).

## Implementation
### Requirements
The following Python libraries are required:
```python
pip install pandas mlxtend plotly networkx
```

### Code Execution
1. Load the dataset:
```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("Market_Basket_Optimisation.csv")
```
2. Preprocess the data:
```python
transactions = df.apply(lambda row: [item for item in row if str(item) != 'nan'], axis=1).tolist()
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)
```
3. Apply the Apriori algorithm:
```python
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
```
4. Visualize the results:
```python
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Bar Chart
fig_bar = px.bar(
    frequent_itemsets.sort_values(by="support", ascending=False).head(10),
    x="itemsets", y="support", title="Top 10 Most Frequently Bought Items"
)
fig_bar.show()
```
```python
# Network Graph
G = nx.DiGraph()
for i, row in rules.iterrows():
    G.add_edge(list(row['antecedents'])[0], list(row['consequents'])[0], weight=row['lift'])

pos = nx.spring_layout(G)
fig_network = go.Figure()
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    fig_network.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines'))
fig_network.show()
```
```python
# Heatmap
heatmap_data = rules.pivot(index='antecedents', columns='consequents', values='lift').fillna(0)
fig_heatmap = px.imshow(heatmap_data, title="Heatmap of Item Associations (Lift Values)")
fig_heatmap.show()
```

## Conclusion
By leveraging association rules, Supermarket X can:
- Identify frequently bought items.
- Implement cross-selling strategies.
- Optimize product placement to increase sales efficiency.

## Author
Developed by Martin. Contact for data science consultation and business intelligence solutions
