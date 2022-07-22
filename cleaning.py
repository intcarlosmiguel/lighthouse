import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

from os.path import exists

def delete_columns(data):
    del data['native_country']
    del data['race']
    del data['education']
    del data['marital_status']
    return data

def graph_bar(data,colun):
    plt.figure(figsize=(16,10))
    coluna = data[colun].values
    coluna = coluna.astype(str)
    a,b = np.unique(coluna,return_counts = True)
    a = np.array(list(map(lambda x:np.char.replace(x, '-', '\n') ,a)))
    a,b = sorting(a,b)
    plt.title(colun)
    plt.bar(a,b)
    plt.show()

def sorting(a,b):
    arr1inds = b.argsort()
    a = a[arr1inds[::-1]]
    b = b[arr1inds[::-1]]
    return a,b

def saldo(df):
    df['saldo'] = df['capital_gain'] - df['capital_loss']
    del df['capital_gain']
    del df['capital_loss']
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df

def histogram(data,coluna):
    hist = []
    count = []
    for i in data[coluna]:
        if(i not in hist):
            hist.append(i)
            count.append(1)
        else:
            index = hist.index(i)
            count[index] += 1
    hist = np.array(hist)
    count = np.array(count)
    hist,count = sorting(hist,count)
    return hist,count

def removequal(data):
    data['fnlwgt'] = data['fnlwgt'].astype(str)
    hist,count = np.unique(data['fnlwgt'],return_counts=True)
    hist,count = sorting(hist,count)
    hist = hist[count > 1]
    for i in hist:
        df = data[data['fnlwgt']==i]
        #display(df)
        majoritary ,_ = histogram(df,"yearly_wage")
        majoritary ,_ = sorting(majoritary ,_)
        majoritary = majoritary[0]
        majoritary = df[df['yearly_wage'] == majoritary].index[1:]
        data = data.drop(index=majoritary)
        #display(data[data['fnlwgt']==hist[0]])
    del data['fnlwgt']
    return data

def generateEncoder(X):
    LE = LabelEncoder()
    a = np.zeros(len(X.T[0]))
    for i in X.T:
        if(type(i[0])!=int):
            teste = LE.fit_transform(i).astype(float)
            a = np.vstack([a, teste])
        else:
            a = np.vstack([a, i.astype(float)])
    a = a[1:,:].T
    X = np.copy(a)
    return X

def MSE(X):
    return (X - np.mean(X.T,axis = 1))/np.std(X.T,axis = 1)

def labelEncoder(data,cont):
    try:
        Y_ = data['yearly_wage'].values
        LE = LabelEncoder()
        Y = np.copy(LE.fit_transform(Y_))
        print(Y_[Y==0][0],0)
        print(Y_[Y==1][0],1)
    except:
        Y = []
    X = data.values[:,:-1]
    X = generateEncoder(X)
    if(cont):
        smote = SMOTE(random_state = 32)
        X, Y = smote.fit_resample(X, Y)
    X = MSE(X)
    return X,Y

def transform(data,cont):
    X,Y = labelEncoder(data,cont)
    label = data['yearly_wage'].unique()
    X,x,Y,y = train_test_split(X, Y, test_size=0.2, random_state=333)
    return X,x,Y,y,label

def ajust(df,alpha):
    L2 = df['yearly_wage'][df['yearly_wage']==' >50K'].index
    L1 = np.array(df['yearly_wage'][df['yearly_wage']==' <=50K'].index)
    np.random.shuffle(L1)
    L = np.hstack((L1[:int((1+alpha)*len(L2))],L2))
    df = df.iloc[L]
    return df

def CM(y_true,y_pred,file,label):
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix = cf_matrix.T/np.sum(cf_matrix,axis = 1)
    cf_matrix = cf_matrix.T
    acc = np.trace(cf_matrix)/np.sum(cf_matrix)
    print(f'Acurácia desse modelo é de: {acc}.')
    plt.figure(figsize = (11.7,8.27))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='flare', fmt='.3g')
    ax.set_title('Matriz de Confusão\n\n');
    ax.set_xlabel('\nValores Preditos')
    ax.set_ylabel('Valores Verdadeiros ');
    ax.xaxis.set_ticklabels(label)
    ax.yaxis.set_ticklabels(label)
    plt.savefig('./img/'+file, dpi=300)
    plt.show()
    #return acc

def bestClassfier(X,Y,x,y,parameters,classfier,NAMEFILE,label,score,showCM):
    if(not exists('./sav/'+NAMEFILE+'.sav')):
        Classfier = GridSearchCV(estimator = classfier(),param_grid=parameters,scoring = score)
        Classfier.fit(X,Y)
        best_pa = Classfier.best_params_
        print(best_pa)
        pickle.dump(classfier(**best_pa),open('./sav/'+NAMEFILE+'.sav','wb'))
    else:
        print('Modelo Carregado!')
        Classfier = pickle.load(open('./sav/'+NAMEFILE+'.sav','rb'))
        Classfier.fit(X,Y)
    previsoes = Classfier.predict(x)
    acc = accuracy_score(y,previsoes)
    if(showCM):
        CM(y,previsoes,NAMEFILE+'.png',label)
    return previsoes,acc
def class_teste(NAMEFILE,X,Y,x):
    Classfier = pickle.load(open('./sav/'+NAMEFILE+'.sav','rb'))
    Classfier.fit(X,Y)
    prev = Classfier.predict(x)
    return prev
