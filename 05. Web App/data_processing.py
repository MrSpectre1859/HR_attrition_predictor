import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um dataframe com as colunas sem tratamento de dados algum
    e retorna um novo df modificado para alimentar o modelo.
      - Se a função encontrar colunas inexistentes (no caso de vir 1 registro),
        é importante garantir que sempre retornemos as MESMAS colunas que 
        o modelo espera, na MESMA ordem.
    """
    
    # Copiamos o DF para não alterar o original
    df_prep = df.copy()
    
    # Mapeamento manual de algumas variáveis
    if 'BusinessTravel' in df_prep.columns:
        df_prep['BusinessTravel'] = df_prep['BusinessTravel'].map({'Travel_Frequently':2, 'Travel_Rarely':1, 'Non-Travel':0})
    if 'Gender' in df_prep.columns:
        df_prep['Gender'] = df_prep['Gender'].map({'Male':1, 'Female':0})
    if 'OverTime' in df_prep.columns:
        df_prep['OverTime'] = df_prep['OverTime'].map({'Yes':1, 'No':0})

    # One-Hot Encoding
    columns_to_dummy = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
    for col in columns_to_dummy:
        if col in df_prep.columns:
            df_prep = pd.get_dummies(df_prep, columns=[col], drop_first=True, dtype='int64')
    
    # Excluir colunas não usadas para treinar o modelo
    cols_to_delete = ['Over18', 'StandardHours', 'EmployeeNumber', 'EmployeeCount', 'Attrition']
    for c in cols_to_delete:
        if c in df_prep.columns:
            df_prep.drop(c, axis=1, inplace=True, errors='ignore')

    
    # Garantir que as colunas fiquem na MESMA ordem que no modelo
    with open("../03. Outputs/features_order.txt", "r") as file:
        expected_order = [line.strip() for line in file.readlines()]

    # Se faltar colunas, vamos criar com o valor igual 0:
    for col in expected_order:
        if col not in df_prep.columns:
            df_prep[col] = 0
    
    df_prep = df_prep[expected_order]
    
    return df_prep

# end