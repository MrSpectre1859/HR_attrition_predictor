# %% [markdown]
# # STEP 1

# %% [markdown]
# ## DATA-DRIVEN DECISIONS FOR HR

# %% [markdown]
# #### IMPORTING THE LIBs AND THE DATASET

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# %%
# Setting the dataframe to show all columns
pd.set_option('display.max_columns', None)

# Ignoring the alerts
warnings.filterwarnings("ignore")

# %%
data = pd.read_csv('../hr_employee_attrition.csv')

# %% [markdown]
# #### UNDERSTANDING THE DATASET

# %%
data.columns

# %%
# To visualize the first 5 rows
data.head()

# %%
# Let's extract some basic info about the total of entries and type of data
data.info()

# %%
# Using the 'describe' we're going to understand better the data distribuition and trends 
data.describe(include='int64')

# %%
# We're doing the same for the Object type of data
data.describe(include='object')

# %%
# Checking for columns with NAN values 
for column in data.columns:
    na = data[column].isna().sum()
    if na > 0:
        print(f"{column} contém {na} valores nulos")
    

# %%
# Checking for columns with duplicated rows
print(data.duplicated().sum())

# %% [markdown]
# #### What are the percentage of attrition?

# %%
# How many people left and how many not
print(data['Attrition'].value_counts())
print(data['Attrition'].value_counts(normalize=True))

# %% [markdown]
# #### How does 'age' impact attrition?

# %% [markdown]
# ##### How is attrition divided by the ages?

# %%
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Age', kde=True, hue='Attrition')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='Attrition', y='Age')
plt.show()

# %% [markdown]
# Elbow method to determine de best number os clusters

# %%
from sklearn.cluster import KMeans

# %%
data['Attrition_numerical'] = data['Attrition'].map({'Yes': 1, 'No': 0})

age_data = data[['Age', 'Attrition_numerical']]

distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(age_data)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('Distortions')
plt.title('Elbow Method results')
plt.show()

# %% [markdown]
# KMEans with 3 clusters (the 'elbow' from the previous image)

# %%
kmeans = KMeans(n_clusters=3, random_state=42)
data['AgeCluster'] = kmeans.fit_predict(age_data)

plt.figure(figsize=(10, 6))
sns.boxplot(x='AgeCluster', y='Age', data=data, hue='Attrition')
plt.title('Age Clusters')
plt.show()

# %% [markdown]
# Using the clusters to analyse satisfaction on the Attrition

# %%
data_temp = data.query("Attrition == 'Yes'")

plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
sns.histplot(data=data_temp, x='EnvironmentSatisfaction', hue='AgeCluster', multiple='dodge', binwidth=0.35).set_xticks(range(1, 5))
plt.title('EnvironmentSatisfaction')

plt.subplot(1, 3, 2)
sns.histplot(data=data_temp, x='JobSatisfaction', hue='AgeCluster', multiple='dodge', binwidth=0.35).set_xticks(range(1, 5))
plt.title('JobSatisfaction')

plt.subplot(1, 3, 3)
sns.histplot(data=data_temp, x='RelationshipSatisfaction', hue='AgeCluster', multiple='dodge', binwidth=0.35).set_xticks(range(1, 5))
plt.title('RelationshipSatisfaction')

plt.tight_layout()
plt.show()


# %% [markdown]
# Analysis with 'YearsAtCompany'

# %%
plt.figure(figsize=(10,6))
sns.kdeplot(data=data, x='YearsAtCompany', hue='Attrition', shade=True)
plt.title('Anos na Empresa vs. Rotatividade')
plt.xlabel('Anos na Empresa')
plt.ylabel('Densidade')
plt.show()

# %% [markdown]
# Analysis with 'OverTime'

# %%
sns.countplot(x='OverTime', hue='Attrition', data=data, palette='magma')
plt.title('Horas Extras vs. Rotatividade')
plt.xlabel('Horas Extras')
plt.ylabel('Contagem')
plt.legend(title='Rotatividade')
plt.show()

# %% [markdown]
# Analysis with 'Department'

# %%
sns.countplot(x='Department', hue='Attrition', data=data, palette='viridis')
plt.title('Departamento vs. Rotatividade')
plt.xlabel('Departamento')
plt.ylabel('Contagem')
plt.legend(title='Rotatividade')
plt.show()

# %% [markdown]
# ---

# %%
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# %%
# =-=-=-=-=-=-=-=-=-=-=-=-=-=
# Pré-processamento dos dados
# =-=-=-=-=-=-=-=-=-=-=-=-=-=

data_prep = data.copy()

data_prep.drop(['Over18', 'StandardHours', 'EmployeeNumber', 'EmployeeCount', 'Attrition'], axis=1, inplace=True)

data_prep['BusinessTravel'] = data_prep['BusinessTravel'].map({'Travel_Frequently' : 2, 'Travel_Rarely' : 1, 'Non-Travel' : 0})
data_prep['Gender'] = data_prep['Gender'].map({'Male' : 1, 'Female' : 0})
data_prep['OverTime'] = data_prep['OverTime'].map({'Yes' : 1, 'No' : 0})

columns_to_dummy = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
for column in columns_to_dummy:
    data_prep = pd.concat([data_prep, pd.get_dummies(data_prep[column], prefix=column, drop_first=True, dtype='int64')], axis=1)

data_prep.drop(columns_to_dummy, axis=1, inplace=True)

data_prep

# %%
# =-=-=-=-=-=-=-=-=
# SMOTE - treino
# =-=-=-=-=-=-=-=-=

X = data_prep.drop(['Attrition_numerical'], axis=1)
y = data_prep['Attrition_numerical']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print('Target var BEFORE SMOTE:', y_train.value_counts())
print('Target var AFTER SMOTE:', y_train_smote.value_counts())

# %%
# =-=-=-=-=-=-=-=-=
# SMOTE - completo
# =-=-=-=-=-=-=-=-=

X = data_prep.drop(['Attrition_numerical'], axis=1)
y = data_prep['Attrition_numerical']

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
print('Target var BEFORE SMOTE:', y.value_counts())
print('Target var AFTER SMOTE:', y_smote.value_counts())

X_train_val, X_test, y_train_val, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# %%
# =================================================
#           DECISION TREE CLASSIFIER
# **SMOTE no dataset completo
# =================================================
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred_dtc_val = dtc.predict(X_val)
print("\nClassification Report: VALIDATION\n", classification_report(y_val, y_pred_dtc_val))

# Classificação - Teste
y_pred_dtc_test = dtc.predict(X_test)
print("\nClassification Report: TEST:\n", classification_report(y_test, y_pred_dtc_test))

# Matriz de Confusão - Teste
cm_test = confusion_matrix(y_test, y_pred_dtc_test)
print("\nConfusion Matrix: TEST:")
print(cm_test)

# %%
# =================================================
#           RANDOM TREE CLASSIFIER
# **SMOTE no dataset completo
# =================================================
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred_rfc_val = rfc.predict(X_val)
print("\nClassification Report: VALIDATION\n", classification_report(y_val, y_pred_rfc_val))

# Classificação - Test
y_pred_rfc_test = rfc.predict(X_test)
print("\nClassification Report: TEST:\n", classification_report(y_test, y_pred_rfc_test))

# Matriz de Confusão - Teste
cm_test = confusion_matrix(y_test, y_pred_rfc_test)
print("\nConfusion Matrix: TEST:")
print(cm_test)

# %%
# =-=-=-=-=-=-=-=-=
# Validação Cruzada
# RFC
# =-=-=-=-=-=-=-=-=

cv_scores = cross_val_score(rfc, X_smote, y_smote, cv=5)
print("Average Accuracy: ", cv_scores.mean())

# %%
y_probs = rfc.predict_proba(X_test)[:, 1]

# Plotando - Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc_score(y_test, y_probs))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()

# %%
# =-=-=-=-=-=-=-=-=-=-=-=-=-=
# AJUSTE 
# Maior precisão na classe 1
# =-=-=-=-=-=-=-=-=-=-=-=-=-=
y_probs = rfc.predict_proba(X_test)[:, 1]
y_pred_adjusted = (y_probs >= 0.6).astype(int)
print(classification_report(y_test, y_pred_adjusted))

cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
print("\nConfusion Matrix with adjusted threshold (0.6):")
print(cm_adjusted)

# %% [markdown]
# ## Exportando o modelo em formato Pickle (.pkl) 

# %%
import joblib
joblib.dump(rfc, 'attrition_pred_model.pkl')

# %% [markdown]
# ---

# %% [markdown]
# # STEP 2

# %% [markdown]
# ## Análisando a importância das features

# %%
# Obter a importância das features
feature_importances = rfc.feature_importances_

# Criar um DataFrame para organizar as features e suas importâncias
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Ordenar as features por importância
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plotar a importância das features
plt.figure(figsize=(10, 8))
plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
plt.xlabel('Importância')
plt.title('Importância das Features para o Modelo RandomForest')
plt.gca().invert_yaxis()  # Inverter para mostrar a maior importância no topo
plt.show()

# %% [markdown]
# ### Análise de Sobrevivência com Cox Proportional Hazards Model

# %%
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(data_prep, duration_col='YearsAtCompany', event_col='Attrition_numerical')

cph.print_summary()

# %% [markdown]
# ## Classificação dos riscos de Attrition

# %%
def classificador_risco(probabilidade):
    if probabilidade < 0.10:
        return "Baixo"
    elif probabilidade < 0.40:
        return "Moderado"
    else:
        return "Alto"


survival_fuctions = cph.predict_survival_function(data_prep, times=[1])
probabilidades = 1 - survival_fuctions.iloc[0, :]

df_probabilidades = data_prep.copy()
df_probabilidades["Probabilidade"] = probabilidades

df_probabilidades['Risco'] = df_probabilidades['Probabilidade'].apply(classificador_risco)

df_probabilidades.head()

# %%
sns.histplot(data=df_probabilidades, x='Risco', hue='Attrition_numerical')

# %%
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

# Plotar as curvas de sobrevivência para cada categoria de risco
plt.figure(figsize=(10, 6))
for categoria in df_probabilidades['Risco'].unique():
    mask = df_probabilidades['Risco'] == categoria
    kmf.fit(data_prep.loc[mask, 'YearsAtCompany'], event_observed=data_prep.loc[mask, 'Attrition_numerical'], label=categoria)
    kmf.plot_survival_function()

plt.title('Curvas de Sobrevivência por Categoria de Risco')
plt.xlabel('Tempo em Anos')
plt.ylabel('Probabilidade de Sobrevivência')
plt.legend(title='Categoria de Risco')
plt.show()


# %% [markdown]
# #### RISK SCORE

# %%
from sklearn.preprocessing import MinMaxScaler


def score_classificator(score):
		if score < 0.2:
			return 'Baixo'
		elif score < 0.5:
			return 'Moderado'
		else:
			return "Elevado"
		

def score_calculator(modelo_cph, df):
	relevant_features = modelo_cph.summary[modelo_cph.summary['p'] < 0.005]
	hazard_ratios = relevant_features['exp(coef)']

	df_risco = df.copy()
	df_risco['Score'] = 0

	for var, hazard_ratio in hazard_ratios.items():
		if var in df_risco.columns:
			df_risco['Score'] += df_risco[var] * hazard_ratio

	scaler = MinMaxScaler()

	df_risco['Score_normalized'] = scaler.fit_transform(df_risco[['Score']])
	
	df_risco['Score_level'] = df_risco['Score_normalized'].apply(score_classificator)

	return df_risco[['Attrition_numerical', 'Score', 'Score_normalized', 'Score_level']]


score_results = score_calculator(cph, data_prep)

# %%
sns.histplot(data=score_results, x='Score', hue='Attrition_numerical')

# %%
from sklearn.metrics import roc_auc_score

# Calcular o C-index (concordância) do modelo Cox
c_index = cph.concordance_index_
print(f"Índice de Concordância (C-index) do Modelo Cox: {c_index:.2f}")

# Preparar os dados para a análise de AUC-ROC
# Vamos usar o 'Score_normalized' como a predição contínua e 'Attrition_numerical' como a variável target
y_true = score_results['Attrition_numerical']
y_scores = score_results['Score_normalized']

# Calcular a AUC-ROC
auc = roc_auc_score(y_true, y_scores)
print(f"AUC-ROC: {auc:.2f}")


# %% [markdown]
# <h1>XGBoost Classifier</h1>

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import shap

model = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_model

# %%
y_pred_prob = best_model.predict_proba(X_val)[:, 1]
y_pred = best_model.predict(X_val)

# %%
auc = roc_auc_score(y_val, y_pred_prob)
print(f'AUC-ROC: {auc:.2f}')
print(classification_report(y_val, y_pred))

# %%
def classificar_risco(probabilidade):
    if probabilidade < 0.10:
        return "Baixo"
    elif probabilidade < 0.40:
        return "Moderado"
    else:
        return "Alto"

df_risco = X_val.copy()
df_risco['Probabilidade'] = y_pred_prob
df_risco['Risco'] = df_risco['Probabilidade'].apply(classificar_risco)
df_risco['Attrition'] = y_val.values
df_risco

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df_risco, x='Risco', hue='Attrition')
plt.title('Quantidade de Pessoas que Saíram (1) x Ficaram (0) / Classificação de Risco')
plt.xlabel('Classificação de Risco')
plt.ylabel('Quantidade de Pessoas')
plt.legend(title='Attrition', labels=['Ficaram (0)', 'Saíram (1)'])
plt.show()

# %%
# Fazer previsões no conjunto de treino
y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
y_train_pred = best_model.predict(X_train)

# Fazer previsões no conjunto de validação
y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
y_val_pred = best_model.predict(X_val)

# Calcular métricas para o conjunto de treino
print("Métricas para o Conjunto de Treinamento:")
print(f"AUC-ROC: {roc_auc_score(y_train, y_train_pred_prob):.2f}")
print(classification_report(y_train, y_train_pred))
print()

# Calcular métricas para o conjunto de validação
print("Métricas para o Conjunto de Validação:")
print(f"AUC-ROC: {roc_auc_score(y_val, y_val_pred_prob):.2f}")
print(classification_report(y_val, y_val_pred))
print()

# Matriz de Confusão para validação
print("Matriz de Confusão para o Conjunto de Validação:")
conf_matrix = confusion_matrix(y_val, y_val_pred)
print(conf_matrix)

# %%
# Plotar Curva ROC para treino e validação
plt.figure(figsize=(12, 6))

# Curva ROC Treinamento
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_prob)
plt.plot(fpr_train, tpr_train, label='Curva ROC - Treinamento', color='blue')

# Curva ROC Validação
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_prob)
plt.plot(fpr_val, tpr_val, label='Curva ROC - Validação', color='red')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para Treinamento e Validação')
plt.legend(loc='lower right')
plt.show()

# %% [markdown]
# ### Ajuste de hiperparâmetros

# %%
model_new = GradientBoostingClassifier(max_depth=1, n_estimators=200, learning_rate=0.2, random_state=42)
model_new.fit(X_train, y_train)

y_train_pred_prob_new = model_new.predict_proba(X_train)[:, 1]
y_train_pred_new = model_new.predict(X_train)
y_val_pred_prob_new = model_new.predict_proba(X_val)[:, 1]
y_val_pred_new = model_new.predict(X_val)

# %%
print("Métricas para o Conjunto de TREINAMENTO (Novo Modelo):")
print(f"AUC-ROC: {roc_auc_score(y_train, y_train_pred_prob_new):.2f}")
print(classification_report(y_train, y_train_pred_new))
print()

print("Métricas para o Conjunto de VALIDAÇÃO (Novo Modelo):")
print(f"AUC-ROC: {roc_auc_score(y_val, y_val_pred_prob_new):.2f}")
print(classification_report(y_val, y_val_pred_new))
print()

print("Matriz de Confusão para o Conjunto de Validação (Novo Modelo):")
conf_matrix_new = confusion_matrix(y_val, y_val_pred_new)
print(conf_matrix_new)

plt.figure(figsize=(12, 6))
fpr_train_new, tpr_train_new, _ = roc_curve(y_train, y_train_pred_prob_new)
plt.plot(fpr_train_new, tpr_train_new, label='Curva ROC - Treinamento (Novo Modelo)', color='green')
fpr_val_new, tpr_val_new, _ = roc_curve(y_val, y_val_pred_prob_new)
plt.plot(fpr_val_new, tpr_val_new, label='Curva ROC - Validação (Novo Modelo)', color='orange')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para Treinamento e Validação (Novo Modelo)')
plt.legend(loc='lower right')
plt.show()

# %%
y_pred_prob_new = model_new.predict_proba(X_val)[:, 1]
y_pred_new = model_new.predict(X_val)

df_risco['Probabilidade'] = y_pred_prob_new
df_risco['Risco'] = df_risco['Probabilidade'].apply(classificar_risco)
df_risco['Attrition'] = y_val.values
df_risco

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df_risco, x='Risco', hue='Attrition')
plt.title('Quantidade de Pessoas que Saíram (1) x Ficaram (0) / Classificação de Risco (NEW)')
plt.xlabel('Classificação de Risco')
plt.ylabel('Quantidade de Pessoas')
plt.legend(title='Attrition', labels=['Ficaram (0)', 'Saíram (1)'])
plt.show()

# %% [markdown]
# ### Explicando as predições usando SHAP

# %%
explainer = shap.Explainer(model_new, X_train)
shap_values = explainer(X_val)
np.shape(shap_values.values)

# %%
shap.plots.waterfall(shap_values[1])

# %%
shap.plots.bar(shap_values)

# %%
shap.plots.beeswarm(shap_values)


