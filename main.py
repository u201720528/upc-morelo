import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score, precision_score, \
    recall_score
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

def ProcesarInput(valorHurto,archivo):
    df = pd.read_csv(archivo, header=None)
    headers = ["numero_cliente", "fecha_inspeccion", "comuna", "distrito", "actividad", "actividad_descripcion",
               "categoria", "giro_suministro", "tarifa", "clave_tarifa", "latitud", "longitud", "tipo_causal",
               "inf_disponible", "sucursal", "fecha_inicio", "fecha_fin", "fecha_creacion", "meses", "f1", "f2", "f3",
               "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19",
               "f20", "f21", "f22", "f23", "f24"
        , "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
               "d18", "d19", "d20", "d21", "d22", "d23", "d24"
        , "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17",
               "c18", "c19", "c20", "c21", "c22", "c23", "c24"]

    df.columns = headers

    df.dropna(subset=["comuna"], axis=0, inplace=True)

    missing_data = df.isnull()
    missing_data.head(5)

    moda_actividad = df["actividad"].mode()[0]
    df["actividad"].replace(0, moda_actividad, inplace=True)

    moda_actividad_desc = df["actividad_descripcion"].mode()[0]
    df["actividad_descripcion"].replace("", moda_actividad_desc, inplace=True)

    # for column in missing_data.columns.values.tolist():
    #     print (missing_data[column].value_counts())
    #     print("")

    df['pendiente'] = 0
    classification = []
    for index in df.iterrows():
        classification.append(valorHurto)
    df['clasificacion'] = classification

    # df.head()
    for index, row in df.iterrows():
        for i in range(24):
            if row[8 + i] == 'N':
                df.loc[index, 'f' + str(i + 1)] = 0
            elif row[8 + i] == '*':
                df.loc[index, 'f' + str(i + 1)] = 1
            elif row[8 + i] == 'U':
                df.loc[index, 'f' + str(i + 1)] = 2
            else:
                df.loc[index, 'f' + str(i + 1)] = 3

    for i in range(24):
        df[["f" + str(i+1)]] = df[["f" + str(i+1)]].astype("int32")

    return df


def CalcularPendiente(df):
    mes = []
    for index in range(24):
        mes.append(index + 1)

    dfc = df.copy()

    print("Fila: " + str(len(dfc.index)))
    for index, row in dfc.iterrows():
        if (index % 1000 == 0):
            print("Van " + str(index))
        consumo = []
        for i in range(24):
            consumo.append(row['c' + str(i + 1)])
        if max(consumo) > 0:
            maximoConsumo = max(consumo)
        for i in range(24):
            dfc.loc[index, 'c' + str(i + 1)] = row['c' + str(i + 1)] / maximoConsumo

        X = []
        Y = []
        for i in range(24):
            X.append(i + 1)
            Y.append(row['c' + str(i + 1)])

        slope, intercept, r, p, std_err = stats.linregress(X, Y)
        if (max(consumo) > 0):
            dfc.loc[index, 'pendiente'] = slope
    return dfc


def LimpiarColumnas(df):
    df.drop("distrito", axis="columns", inplace=True)
    df.drop("comuna", axis="columns", inplace=True)
    df.drop("categoria", axis="columns", inplace=True)
    df.drop("actividad", axis="columns", inplace=True)
    df.drop("actividad_descripcion", axis="columns", inplace=True)
    df.drop("giro_suministro", axis="columns", inplace=True)
    df.drop("latitud", axis="columns", inplace=True)
    df.drop("longitud", axis="columns", inplace=True)
    df.drop("clave_tarifa", axis="columns", inplace=True)
    df.drop("tipo_causal", axis="columns", inplace=True)
    df.drop("inf_disponible", axis="columns", inplace=True)
    df.drop("sucursal", axis="columns", inplace=True)
    df.drop("fecha_creacion", axis="columns", inplace=True)
    df.drop("fecha_fin", axis="columns", inplace=True)
    df.drop("fecha_inicio", axis="columns", inplace=True)
    df.drop("fecha_inspeccion", axis="columns", inplace=True)
    df.drop("numero_cliente", axis="columns", inplace=True)
    df.drop("tarifa", axis="columns", inplace=True)
    df.drop("meses", axis="columns", inplace=True)
    for i in range(24):
        df.drop("c" + str(i + 1), axis="columns", inplace=True)

    for i in range(24):
        df.drop("d" + str(i + 1), axis="columns", inplace=True)

def Main():
    print("Procesando primer archivo (hurtos)")
    url = "hurtos.csv"
    primeraData = ProcesarInput(1, url)

    print("Procesando segundo archivo (no hurtos)")
    url = "no_hurtos.csv"
    segundaData = ProcesarInput(0, url)

    print("Juntando")
    frames = [primeraData, segundaData]
    MainData = pd.concat(frames, ignore_index=True)
    print("Cantidad de datos cargados: " + str(MainData.shape[0]))

    dfDistrito = pd.get_dummies(MainData["distrito"])
    dfActividad = pd.get_dummies(MainData["actividad_descripcion"])
    dfTarifa = pd.get_dummies(MainData["tarifa"])
    dfSucursal = pd.get_dummies(MainData["sucursal"])
    dfGiro = pd.get_dummies(MainData["giro_suministro"])

    MainData = pd.concat([MainData, dfDistrito], axis=1, join="inner")
    MainData = pd.concat([MainData, dfActividad], axis=1, join="inner")
    MainData = pd.concat([MainData, dfTarifa], axis=1, join="inner")
    MainData = pd.concat([MainData, dfSucursal], axis=1, join="inner")
    MainData = pd.concat([MainData, dfGiro], axis=1, join="inner")

    print("Calculando pendiente")
    MainData = CalcularPendiente(MainData)
    print("Pendiente calculada")

    print("Eliminando columnas")
    LimpiarColumnas(MainData)

    MainData.to_csv("MainData.csv", mode='w', index=False)

    print(MainData.head())
    print(MainData.tail())

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta función muestra y dibuja la matriz de confusión.
    La normalización se puede aplicar estableciendo el valor `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalización')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

def LogisticRegressionModel():
    ModelData = pd.read_csv("MainData.csv")
    print(ModelData.head())
    ModelData['clasificacion'] = ModelData['clasificacion'].astype('int')

    X = np.array(ModelData.drop(['clasificacion'], 1))
    y = np.array(ModelData['clasificacion'])
    f = X.shape
    #print(f)

    X = preprocessing.StandardScaler().fit(X).transform(X)

    validation_size = 0.15
    seed = 4

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

    print('Train set:', X_train.shape, Y_train.shape)
    print('Test set:', X_test.shape, Y_test.shape)

    cantidadMuestra = X_train.shape[0] + X_test.shape[0]
    cantidadEntrenamiento = X_train.shape[0]
    cantidadTest = X_test.shape[0]

    print("Cantidad Muestra: " + str(cantidadMuestra))
    print("Cantidad Entrenamiento: " + str(cantidadEntrenamiento))
    print("Cantidad Test: " + str(cantidadTest))

    #C=1.0
    model = linear_model.LogisticRegression(C=99999999, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False)
    model.fit(X_train, Y_train)
    #print(model)

    predictions = model.predict(X_test)
    #var = predictions[0:5]
    #print(model.score(X, y))

    prescision = precision_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions)

    print("Indice Jaccard: " + str(jaccard_score(Y_test, predictions)))
    print("Accuracy(Exactitud): " + str(accuracy_score(Y_test, predictions)))
    print("Precision: " + str(precision_score(Y_test, predictions)))
    print("Recall(Sesibilidad): " + str(recall_score(Y_test, predictions)))
    print("F1: " + str(2*(recall * prescision) / (recall + prescision)))
    # Save Model
    print("Saving Model")
    pickle.dump(model, open("lds_model.sav", "wb"))
    print("Save Model")
    print(cross_val_score(model, X_train, Y_train, cv=5))

    # Calcular la matriz de confusión
    cnf_matrix = confusion_matrix(Y_test, predictions, labels=[1, 0])

    np.set_printoptions(precision=2)
    # Dibujar la matriz de confusión no normalizada
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Matriz de confusión')

    #ProbarModelo(X_train, Y_train)

def Test():
    df = pd.read_csv("MainData.csv")
    missing_data = df.isnull()
    print(missing_data.head(5))
    for column in missing_data.columns.values.tolist():
        print (missing_data[column].value_counts())
        print("")

def ProbarModelo(X, y):
    #loaded_model = pickle.load(open('lds_model.sav', 'rb'))
    classifiers = []
    classifiers_titles = ["clf_overfit_1", "clf_overfit_2", "clf_overfit_3", "clf_right_fit", "clf_underfit_1",
                          "clf_underfit_2", "clf_underfit_3"]

    clf_overfit_1 = linear_model.LogisticRegression(random_state=0, C=99999999999, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False).fit(X, y)
    clf_overfit_2 = linear_model.LogisticRegression(random_state=0, C=10000000, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False).fit(X, y)
    clf_overfit_3 = linear_model.LogisticRegression(random_state=0, C=1000, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False).fit(X, y)
    clf_right_fit = linear_model.LogisticRegression(random_state=0, C=1, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False).fit(X, y)
    clf_underfit_1 = linear_model.LogisticRegression(random_state=0, C=0.001, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False).fit(X, y)
    clf_underfit_2 = linear_model.LogisticRegression(random_state=0, C=0.0001, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False).fit(X, y)
    clf_underfit_3 = linear_model.LogisticRegression(random_state=0, C=0.00001, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False).fit(X, y)

    classifiers = np.append(classifiers, clf_overfit_1)
    classifiers = np.append(classifiers, clf_overfit_2)
    classifiers = np.append(classifiers, clf_overfit_3)
    classifiers = np.append(classifiers, clf_right_fit)
    classifiers = np.append(classifiers, clf_underfit_1)
    classifiers = np.append(classifiers, clf_underfit_2)
    classifiers = np.append(classifiers, clf_underfit_3)

    clf_df = pd.DataFrame()
    clf_df["Classifier_object"] = classifiers
    clf_df["Classifier"] = classifiers_titles
    w_vectors = np.empty((0, 2))
    for clf in classifiers:
        w_vectors = np.append(w_vectors, clf.coef_, axis=0)

    clf_df["w_0"] = w_vectors[:, 0]
    clf_df["w_1"] = w_vectors[:, 1]

    print(clf_df)

#Main()
LogisticRegressionModel()

#Test()