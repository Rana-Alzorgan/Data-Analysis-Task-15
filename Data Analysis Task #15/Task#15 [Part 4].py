# هل يمكن استخدم وزن الالماسه لتوقع سعرها
#%%
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

diamonds = sns.load_dataset('diamonds')
diamonds.head

x=diamonds[['carat']]
y=diamonds['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model_diamond = LinearRegression()
model_diamond.fit(x_train,y_train)

y_pred = model_diamond.predict(x_test)
r2 = model_diamond.score(x_test,y_test)

comparison = pd.DataFrame({'Actual Price': y_test,'Predicted Price':y_pred})
print(comparison)
print("R^2 = ",r2)
# R^2 = 0.84893 وهذا يعني ان الوزن يفسر 85% تقريبا من التباين بالسعر

print("Slope =", model_diamond.coef_)
print("Intercept =", model_diamond.intercept_)

# كل زيادة بمقدار 1 carat تؤدي الى زيادة مقدارها حوالي 7768.91 في السعر

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.scatter(x_test['carat'],y_test,alpha=0.3,label='Actual Data')

sorted_data = x_test.copy()
sorted_data['predicted_price']=model_diamond.predict(sorted_data[['carat']])
sorted_data=sorted_data.sort_values('carat')
plt.plot(sorted_data['carat'],sorted_data['predicted_price'],label='Regression Line')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Diamond Carat vs Price')
plt.legend()
plt.show()
