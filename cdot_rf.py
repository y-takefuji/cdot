import pandas as pd,numpy as np
train = pd.read_csv('train.csv')
X_train = train.drop(['project_number','bid_days','bid_total','engineers_estimate','start_date'],axis=1)
y_train = train['bid_total']
test = pd.read_csv('test.csv')
X_test = test.drop(['project_number','bid_days','bid_total','engineers_estimate','start_date'],axis=1)
y_test = test['bid_total']

from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(n_estimators=82, n_jobs=-1,min_samples_split=2,random_state=90)
clf.fit(X_train,y_train)
print("p=",len(X_train.columns),"accuracy=",clf.score(X_test,y_test))
importances=clf.feature_importances_
import matplotlib.pyplot as plt
'''
plt.yscale('log', nonposy='clip')
plt.hist(importances.ravel(), bins=2048, range=(0.0, 1e-1), fc='k', ec='k')
plt.title("cdot feature importances")
plt.show()
'''
dic=dict(zip(X_train.columns,clf.feature_importances_))
i=0
labels=[[]]
clf=RandomForestRegressor(n_estimators=82, n_jobs=-1,min_samples_split=2,random_state=90)
x=[]
y=[]
f=open('x_yRF.csv','bw')
for p in range(20,1000,1):
 for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
  labels=np.column_stack((labels,[item[0]]))
  i=i+1
  if i==p:
   break
 xx=X_train[labels[0][0]]
 xt=X_test[labels[0][0]]
 for i in range(1,p-1):
  xx=np.column_stack((xx,X_train[labels[0][i]]))
  xt=np.column_stack((xt,X_test[labels[0][i]]))
 clf.fit(xx,y_train)
 accuracy=clf.score(xt,y_test)
# print("p=",p," accuracy=",accuracy)
 st=str(p)+','+str("%.6f"%accuracy)+'\n'
 f.write(st.encode('utf-8'))
 x.append(int(p))
 y.append(float(accuracy))
plt.xlabel('no. of parameters')
plt.ylabel('accuracy')
plt.plot(x,y,'ko')
fig=plt.figure(1)
plt.savefig('accuracy.png',dpi=fig.dpi,bbox_inches='tight')
plt.show()
f.close()

