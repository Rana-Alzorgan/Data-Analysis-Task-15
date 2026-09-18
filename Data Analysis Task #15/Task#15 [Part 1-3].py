#=============================
# PART 1
#=============================

#%% 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

tips = sns.load_dataset('tips')
tips.head

# detect x and y
x= tips[['total_bill']]
y= tips['tip']

# split data 20/80
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# model learninig
model = LinearRegression()
model.fit(x_train,y_train)

# prediction
y_pred = model.predict(x_test)

# first 10 predction vs real data
comparission = pd.DataFrame({'Actual Tip':y_test , 'Predicted Tip ': y_pred})
print(comparission.head(10))

# R^2
r2 = model.score(x_test,y_test)
print("R^2 = ",r2)

# قيمة R² هي 0.54، وهذا يعني أن قيمة total_bill تفسّر حوالي 54% من التباين في قيمة tip، بينما النسبة المتبقية (46%) قد تكون مرتبطة بعوامل أخرى لم يتم تضمينها في النموذج

#=============================
# PART 2
#=============================
#%% 
slope = model.coef_[0]
intercept = model.intercept_
print("slope = " ,slope)
print("Intercept = ",intercept)
#كلما زادت قيمة الفاتورة بمقدار دولار واحد، فإن الإكرامية المتوقعة تزيد بحوالي 0.107 دولار، في المتوسط

#%%
x_size= tips[['size']]
y= tips['tip']
x_train_size,x_test_size,y_train_size,y_test_size = train_test_split(x_size,y,test_size=0.2,random_state=42)
model_size =LinearRegression()
model_size.fit(x_train_size,y_train_size)
size_r2=model_size.score(x_test_size,y_test_size)
print("Size R^2 = ",size_r2)


# total_bill هو مؤشر اقوى على الاكرامية
# عدد الاشخاص على الطاولة له علاقة بالاكرامية لكن العلاقة ليست قوية بما يكفي للاعتماد على عدد الاشخاص وحده لتوقع قيمة الاكرامية بدقة
# النموذج الذي استخدم الحجم حقق R^2 اقل بكثير من الذي استخدم total_bill و هذا يعني ان عدد الاشخاص يفسر جزء محدود من التباين في قيمة الاكرامية
# لذلا لا يمكننا القول ان الطاولات الكبيرة تترك اكرامية اعلى بشكل مؤكد
# من الافضل اخذ قيمة الفاتورة و عوامل اخرى بعين الاعتبار عند تحليل او توقع الاكرامية

#=============================
# PART 3
#=============================
#%%

flights = sns.load_dataset('flights')
flights.head()

flights['month_number']=range(1,len(flights)+1)
x = flights[['month_number']]
y= flights['passengers']
trend_model = LinearRegression()
trend_model.fit(x,y)

last_month = flights['month_number'].max()
print(last_month)
future_months = pd.DataFrame({"month_number":range(last_month+1,last_month+7)})
print(future_months)
future_predictions = trend_model.predict(future_months)
future_predictions
forecast = future_months.copy()
forecast['predicted_passengers'] = future_predictions
forecast

#%%
import matplotlib.pyplot as plt

# توقعات النموذج لكل البيانات الأصلية
trend_predictions = trend_model.predict(flights[['month_number']])
plt.figure(figsize=(12, 6))
# البيانات الأصلية
plt.scatter(flights['month_number'],flights['passengers'],label='Actual Data')

# خط الاتجاه
plt.plot(flights['month_number'],trend_predictions,label='Trend Line')

# التوقعات المستقبلية
plt.plot(future_months['month_number'],future_predictions,marker='o',label='6-Month Forecast')

plt.xlabel('Month Number')
plt.ylabel('Passengers')
plt.title('Linear Regression Trend and 6-Month Forecast')

plt.legend()
plt.show()

# نلاحظ ان البيانات الاصليه ليست خط مستقيم فهي تحتوي على صعود و نزول متكرر عبر الشهور و هذا ما نسميه الموسمية
# ال Linear Regression يحاول ايجاد خط مستقيم واحد لكن بياناتنا فيها نمط صعود و نزول يتبع للموسمية بالاضافه الى وجود نمط عام صاعد
# يعني هو يجمع بين ال Trend & Sesonality لكن ال Linear regression لأا يستطيع التقاط ال seasonality هو فقط يلتقط ال trand
# عند مقارنة توقعات linear Regression مع Moving Average على نفس البيانات فان linear Regression يلتقط الاتجاه العام للبيانات و يظهر الزيادة التدريجية في التوقعات
# بينما يعتمد Moving Average على القي الاخيرة و يكون اكتر ارتباط بالمستوى الحالي للبيانات 
# مع وجود نمط موسمي قوي فان كلا النموذجين لا يمثل الموسمية بشكل جيد لذلك قد تختلف التوقعات عن القيم الفعليه في الشهر التي ترتفع او تنخفض موسميا
# بالتالي للتعامل مع هذه البيانات بشكل افضل الافضل استخدام Seasonal Naive لانه يعتمد على قيمة الفترة الموسمية السابقة
