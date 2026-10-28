### LIBRERIE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches 


###############################################################################
#                               FUNZIONI                                      #
###############################################################################

## FUNZIONE PER CONVERTIRE IN DATE 

def converti_data(data):
    try:
        if pd.isna(data):
            return pd.NaT
        return pd.to_datetime(str(int(data)), format="%Y%m%d")
    except:
        return pd.NaT
    
## FUNZIONE DI MAPPATURA SCORE SONEPAR

def mappa_sonepar(score):
    if pd.isna(score) or score == 'ND':
        return np.nan
    
    # Estraiamo il numero (Rischio)
    rischio = int(score[0]) # Prende 1, 2, 3...
    
    # Estraiamo la lettera (Trend)
    # Assegniamo un piccolo peso: A=+0.2, N=0.0, B=-0.2
    trend = score[1] if len(str(score)) > 1 else 'N'
    modificatore = {'A': 0.2, 'N': 0, 'B': -0.2, 'P': -0.5}.get(trend, 0)
    
    # Calcoliamo un punteggio invertito (perché 1 è il migliore)
    # Scala finale: circa da 7 (ottimo) a 0 (pessimo)
    return (8 - rischio) + modificatore

## QUESTA FUNZIONE PERMETTE DI CORREGGERE LE DECISIONI DEL MODELLO SULLA BASE DELLO
## STATO DELL'ATTIVITA DELL'AZIENDA RICHIEDENTE E DELLA SUA NATURA GIURIDICA. LE
## SOGLIE FISSATE DOVREBBERO ESSERE DATE DA SONEPAR IN QUANTO SONO LORO CHE DEVONO
## DECIDERE QUANTO PENALIZZARE IL FATTO DI ESSERE UNA DITTA INDIVIDUALE RISPETTO
## AD UNA SOCIETA DI CAPITALI, ECC...

def motore_decisionale(row):
    # 1. Recupero dati necessari
    # Usiamo .get() per sicurezza se la colonna non fosse presente in tutte le fasi
    stato = str(row.get('STATO_ATTIVITA', 'A')).upper().strip() 
    prob = row['Prob_Modello_SI']
    natura = row.get('SINTESI_NATURA_GIURIDICA', 'CAPITALI')
    
    # 2. FILTRI BLOCCANTI (KO)
    # 'P' qui assorbe il concetto di Procedure Concorsuali
    if stato in ['I', 'R', 'S', 'C', 'P']:
        descrizione_ko = {
            'I': 'Inattiva', 'R': 'Receduta', 'S': 'Sospesa', 
            'C': 'Cessata', 'P': 'Procedure Concorsuali'
        }
        motivo = descrizione_ko.get(stato, stato)
        return f"RIFIUTATO (KO: {motivo})"
    
    # 3. SOGLIE DINAMICHE PER NATURA GIURIDICA
    soglie = {
        'CAPITALI': 0.68,
        'PERSONE': 0.75,
        'INDIVIDUALE': 0.88,
        'ENTI_COOP_ALTRO': 0.80
    }
    
    soglia_target = soglie.get(natura, 0.80)
    
    # 4. DECISIONE FINALE
    if prob >= soglia_target:
        return "APPROVATO"
    else:
        return f"RIFIUTATO (Score {prob:.2f} < Soglia {soglia_target})"

# NATURA_GIURIDICA (mappatura)

def map_natura_precisa(sigla):
    sigla = str(sigla).upper().strip()
    capitali = ['SR', 'SP', 'SU', 'RS'] # Srl, Spa, Srl Unipers., Srls
    persone = ['SN', 'AS', 'SO']        # Snc, Associazioni, Società di fatto
    individuali = ['DI', 'AU']          # Ditta Indiv., Autonomo
    # Il resto: SC (Coop), EN (Ente), SE (Semplificata), FO (Fondazione), CC, SL
    
    if sigla in capitali: return 'CAPITALI'
    if sigla in persone: return 'PERSONE'
    if sigla in individuali: return 'INDIVIDUALE'
    return 'ENTI_COOP_ALTRO'

###############################################################################
#                            INIZIO ANALISI                                   #
###############################################################################

# CARICAMENTO DATASET
df = pd.read_excel("DatabaseProposteFido_202602.xlsx", index_col="ID")

# Verifichiamo le dimensioni del dataset (righe, colonne)
print("Dimensioni dataset originale:", df.shape) # (5857, 24-1)

# Controllo dei tipi di variabili presenti nel dataset
print("\nTipi di variabili df:")
print(df.dtypes)

# Valori MANCANTI
df_missing = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing %': df.isnull().mean() * 100
}) #.sort_values(by='Missing %', ascending=False) 

# Verifico le osservazioni con valori mancanti
righe_CCIAnull = df[df['CCIAA_IMPRESA'].isna()]
righe_DataIAnull = df[df['DATA_INZIO_ATTIVITA'].isna()]

# Osservo le imprese Inattive, Resgistrate,Sospese, con procedure e cancellate

# ----- NB: Le imprese Registrate nessuna ha la data di inizio come le inattive ---- #

righe_ImpInattive = df[df['STATO_ATTIVITA']== 'I']
righe_ImpRegistrate = df[df['STATO_ATTIVITA']== 'R']
righe_ImpSospese = df[df['STATO_ATTIVITA']== 'S']
righe_ImpCessata = df[df['STATO_ATTIVITA']== 'C']
righe_ImpProcedure = df[df['STATO_ATTIVITA']== 'P']

# Controllo DUPLICATI ed eliminazione
df_duplicati = df.duplicated().sum()
print("Numero righe duplicate:", df_duplicati) # 18
df = df.drop_duplicates()


# MATRICE CORRELAZIONE df delle sole variabili NUMERICHE 
    
# Includiamo int64 e float64, escludiamo variabili categoriche o temporali non trasformate

df_numeric = df.select_dtypes(include=[np.number])

# Calcolo della matrice di correlazione (Pearson)
# Pandas utilizza automaticamente osservazioni pairwise per gestire eventuali valori mancanti

df_corr_matrix = df_numeric.corr(method='pearson')


###############################################################################
#                          DATA PREPARATION (df_prep)                         #
###############################################################################

# Creiamo una copia per non sporcare il df originale

df_prep = df.copy()

print("\nTipi di variabili df_prep:")
print(df_prep.dtypes)

# Modifico i nomi delle variabili Macroeconomiche

df_prep['inflazione']= df["Aspettative Inflazione Italia a 12 mesi (fonte: Banca d'Italia)"]
df_prep['pil']= df["Crescita attesa PIL Italia per l'anno 2024 (Fonte: ISTAT)"]

# Rimozione colonne ridondanti, costanti o leakage

colonne_da_escludere = ["Fascia Fido", "DATA_ISCRIZIONE", "SCORE_SONEPAR",
                        "CLASSE_SCORE_INNOLVA","BRAND_SONEPAR","CCIAA_IMPRESA", "CODICE_ATECO_NP",
                        "FATTURATO","CODICE_ATECO","SCORE_INNOLVA",
                        "Aspettative Inflazione Italia a 12 mesi (fonte: Banca d'Italia)",
                        "Crescita attesa PIL Italia per l'anno 2024 (Fonte: ISTAT)"]

df_prep = df_prep.drop(columns=colonne_da_escludere)


# Valori MANCANTI
df_prep_missing = pd.DataFrame({
    'Missing Count': df_prep.isnull().sum(),
    'Missing %': df_prep.isnull().mean() * 100
}) #.sort_values(by='Missing %', ascending=False)


###############################################################################
#                    CREAZIONE ETA IMPRESA IN GIORNI                          #
###############################################################################

# Conversione delle colonne data
df_prep["DATA_CALCOLO"] = df_prep["DATA_CALCOLO"].apply(converti_data)
df_prep["DATA_INZIO_ATTIVITA"] = df_prep["DATA_INZIO_ATTIVITA"].apply(converti_data)

# Gestione valori nulli in DATA_INZIO_ATTIVITA 
# Imprese Inattive (I) o Registrate (R) con DATA_INZIO_ATTIVITA mancante → età = 0
# Utilizziamo .isin(['I', 'R']) per maggiore pulizia se decidessi di aggiungere altri stati in futuro

mask_nuove_imprese = (df_prep["STATO_ATTIVITA"].isin(['I', 'R'])) & (df_prep["DATA_INZIO_ATTIVITA"].isna())

# Imputazione: usiamo la DATA_CALCOLO come data inizio per azzerare l'anzianità
df_prep.loc[mask_nuove_imprese, "DATA_INZIO_ATTIVITA"] = df_prep.loc[mask_nuove_imprese, "DATA_CALCOLO"]

# Pulizia: eliminare osservazioni dove DATA_INZIO_ATTIVITA rimane nulla per altri STATO_ATTIVITA (es. Attive senza data)
df_prep = df_prep[df_prep["DATA_INZIO_ATTIVITA"].notna()]

# Calcolo età dell’impresa in giorni
df_prep["ETA_IMPRESA_GIORNI"] = (df_prep["DATA_CALCOLO"] - df_prep["DATA_INZIO_ATTIVITA"]).dt.days

# Trasformazione logaritmica
df_prep["LOG_ETA_IMPRESA"] = np.log1p(df_prep["ETA_IMPRESA_GIORNI"])  # log(1 + x) per evitare log(0)


#------------------------------------------------------------------------------------------------#
# PER I GRAFICI DELLE DISTRIBUZIONI DI ETA IMPRESA E LOG ETA IMPRESA ANDARE IN FONDO ALLO SCRIPT #
#------------------------------------------------------------------------------------------------#


# Eliminazione colonne originali

df_prep = df_prep.drop(columns=["DATA_CALCOLO", "DATA_INZIO_ATTIVITA"])


# - Feature Engineering: ITP
df_prep = df_prep[df_prep['ITP'] != -999] # eliminazione errori digitazione


# Opzione 1: eliminare righe con NaN
df_prep = df_prep[df_prep['ITP'].notna()]


'''

# Opzione 2: imputare NaN con 1000 
df_prep['ITP'] = df_prep['ITP'].fillna(1000)


# Creazione colonna flag missing (opzionale, utile per ML)

df_prep['ITP_MISSING'] = df_prep['ITP'].isna().astype(int)

'''

# Trasformazione logaritmica (log1p per evitare problemi con 0)
df_prep['LOG_ITP'] = np.log1p(df_prep['ITP']) 

#------------------------------------------------------------------------------------------------#
#       PER I GRAFICI DELLE DISTRIBUZIONI DI ITP E LOG ITP ANDARE IN FONDO ALLO SCRIPT           #
#------------------------------------------------------------------------------------------------#

# Trasformazione LOGARITMICA di DIPENDENTI e NUMERO_IMMOBILI

for col in ["DIPENDENTI","NUMERO_IMMOBILI"]:
    df_prep[f"LOG_{col.replace(' ', '_')}"] = np.log1p(df_prep[col])
    
#------------------------------------------------------------------------------------------------#
#       PER I GRAFICI DELLE DISTRIBUZIONI DI NUMERO DIPENDENTI, LOG NUMERO DIPENDENTI            #
#       , NUMERO IMMOBILI E LOG NUMERO IMMOBILI ANDARE IN FONDO ALLO SCRIPT                      #
#------------------------------------------------------------------------------------------------#


###############################################################################
#                   CODIFICA DELLE VARIABILI CATEGORICHE                      #
###############################################################################

# Esito_finale (variabile TARGET: binaria in 'SI': 1, 'NO': 0)

df_prep["Esito_finale"] = df_prep["Esito_finale"].map({'SI': 1, 'NO': 0})

# Mappiamo la natura giuridica delle aziende richiedenti con la funzione di mappatura

df_prep['SINTESI_NATURA_GIURIDICA'] = df_prep['NATURA_GIURIDICA'].apply(map_natura_precisa)

# Visualizziamo la distribuzione per capire quante aziende ci sono in ogni categoria

print(df_prep['SINTESI_NATURA_GIURIDICA'].value_counts())

# Selezione finale colonne per Modellazione

colonne_finali = [
    "Esito_finale", "VALORE FIDO RICHIESTO","LOG_DIPENDENTI",
    "LOG_ITP","LOG_NUMERO_IMMOBILI",
    "REVISIONE","AFFIDATA","LOG_ETA_IMPRESA","inflazione","pil"]

df_model = df_prep[colonne_finali]

###############################################################################
#                          VERIFICA DEL VIF                                   #
###############################################################################

# 1. Definiamo X e y senza aggiungere costanti per ora

X = df_model.iloc[:, 1:]
y = df_model.iloc[:, 0] # Usiamo Series invece di DataFrame per evitare il warning ravel()

# 2. Verifica allineamento indici (fondamentale!)

if not X.index.equals(y.index):
    print("ATTENZIONE: Indici disallineati! Eseguo il riallineamento.")
    X, y = X.align(y, join='inner', axis=0)
 
# Calcolo VIF per verifica
X_vif = add_constant(X)
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print(vif_data)

corr_matrix = df_model.corr()


###############################################################################
#                        TRAINING DEL MODELLO RIDGE                           #
###############################################################################

# 1. PRIMO SPLIT: Creazione Holdout Set (20% del totale)
# Usiamo stratify=y per mantenere la proporzione di SI/NO

X, X_holdout, y, y_holdout = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------La configurazione migliore è std e 0.30 e robust ------------ #


# ------------------ PARAMETRI DI CONFIGURAZIONE ---------------------------- #
test_size_val = 0.30      # Puoi variare lo split (es. 0.20, 0.25, 0.30)
scaling_method = 'robust'    # Opzioni: 'std' (Standard), 'minmax' (MinMax), 'robust' (Robust)
# --------------------------------------------------------------------------- #

# Generazione dataset

# Split dinamico
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_val, random_state=42, stratify=y
)

# Normalizzazione variabile
scalers = {
    'std': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

# Identifica le colonne da scalare e quelle da mantenere originali(Alcune variabili variano già tra 0 e 1)

cols_to_scale = ["VALORE FIDO RICHIESTO", "LOG_DIPENDENTI", "LOG_ITP", "LOG_NUMERO_IMMOBILI", "LOG_ETA_IMPRESA"]
cols_to_keep = ["REVISIONE", "AFFIDATA", "inflazione", "pil"]

# Fit dello scaler solo sulle colonne selezionate del Training Set

scaler = StandardScaler()

X_train_scaled_part = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled_part = scaler.transform(X_test[cols_to_scale])

# Ricostruzione del set completo (Scaled + Originali)

X_train_final = np.hstack([X_train_scaled_part, X_train[cols_to_keep].values])
X_test_final = np.hstack([X_test_scaled_part, X_test[cols_to_keep].values])

# Fit del modello (Regressione Logistica con Ridge/L2)

ridge_log_cv = LogisticRegressionCV(
    Cs=10, 
    cv=5, 
    penalty='l2', 
    scoring='accuracy', # Potresti cambiarlo in 'recall' visto il business case 
    random_state=42,
    max_iter=1000
)

ridge_log_cv.fit(X_train_final, y_train)

# 5. Output dei risultati

print(f"--- Configurazione: Split={test_size_val}, Scaler={scaling_method} ---")
print(f"Miglior C trovato: {ridge_log_cv.C_[0]:.4f}")
print(f"Accuratezza Test: {ridge_log_cv.score(X_test_final, y_test):.4f}")

###### Miglior C trovato con nomralizzazione Z_score e split 70/30: 0.0464
###### Accuratezza Test Z_score e split 70/30: 0.8376

###### Miglior C trovato con nomralizzazione Z_score e split 75/25: 0.0464
###### Accuratezza Test Z_score e split 75/25: 0.8373

###### Miglior C trovato con nomralizzazione Z_score e split 80/20: 0.3594
###### Accuratezza Test Z_score e split 80/20: 0.8352

###### Miglior C trovato con nomralizzazione minmax e split 70/30: 0.0464
###### Accuratezza Test minmax e split 70/30: 0.8376

###### Miglior C trovato con nomralizzazione minmax e split 75/25: 0.0464
###### Accuratezza Test minmax e split 75/25: 0.8373

###### Miglior C trovato con nomralizzazione minmax e split 80/20: 0.3594
###### Accuratezza Test minmax e split 80/20: 0.8352

###### Miglior C trovato con nomralizzazione minmax e split 70/30: 0.0464
###### Accuratezza Test robust e split 70/30: 0.8376

###### Miglior C trovato con nomralizzazione minmax e split 75/25: 0.0464
###### Accuratezza Test robust e split 75/25: 0.8373

###### Miglior C trovato con nomralizzazione robust e split 80/20: 0.3594
###### Accuratezza Test robust e split 80/20: 0.8352

##############################################################################
#                  GENERAZIONE DELLE LEARNING CURVES                         #
##############################################################################

# Utilizziamo il miglior parametro C trovato nelle tue prove (es. 0.0464)
# e il modello di regressione logistica con regolarizzazione L2 (Ridge)

model = LogisticRegression(C=0.0464, penalty='l2', solver='lbfgs', max_iter=1000)

# Definiamo le dimensioni dei subset di training (da 10% a 100% del dataset)
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_final, y_train.values.flatten(), cv=5, 
    scoring='accuracy', n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42
)

# Calcoliamo medie e deviazioni standard dei punteggi
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

#-----------------------------------------------------------------------------#
#      PER GRAFICO LEARNING CURVES ANDARE IN FONDO AL CODICE                  #
#-----------------------------------------------------------------------------#


###############################################################################
#                     VALIDAZIONE FINALE E SIGNIFICATIVITÀ                    #
###############################################################################

# Preparazione coerente dell'Holdout (stesso ordine del Training!)

X_holdout_scaled_part = scaler.transform(X_holdout[cols_to_scale])
X_holdout_final = np.hstack([X_holdout_scaled_part, X_holdout[cols_to_keep].values])

# APPLICA IL MODELLO ALL'HOLDOUT (senza ri-addestrare)
# Usiamo .score() o .predict() per vedere se ha imparato bene 

accuratezza_reale = ridge_log_cv.score(X_holdout_final, y_holdout)


# Creiamo un DataFrame per visualizzarli insieme ai nomi delle colonne
# Assicurati che 'features_names' contenga i nomi delle tue variabili dopo il preprocessing
# Estrazione dei coefficienti dal modello addestrato

coefficients = ridge_log_cv.coef_[0]

features_names = cols_to_scale + cols_to_keep

importance_df = pd.DataFrame({
    'Variabile': features_names,
    'Coefficiente (Peso)': coefficients,
    'Assoluto': np.abs(coefficients)
})

# Calcoliamo l'Odds Ratio 
# Ricorda: Odds Ratio = exp(coefficiente)

importance_df['Odds_Ratio'] = np.exp(importance_df['Coefficiente (Peso)'])

print(df_prep.groupby('Esito_finale')['ETA_IMPRESA_GIORNI'].median())

# Ordiniamo per importanza assoluta per vedere cosa pesa di più
importance_df = importance_df.sort_values(by='Assoluto', ascending=False)

print("--- SIGNIFICATIVITÀ E PESI DEL MODELLO RIDGE ---")
print(importance_df[['Variabile', 'Coefficiente (Peso)', 'Odds_Ratio']])


# Otteniamo le probabilità per la classe "SI" (indice 1)

y_probs_holdout = ridge_log_cv.predict_proba(X_holdout_final)[:, 1]

# Genera le predizioni sull'Holdout Set (dati mai visti prima)
# Impostiamo una soglia più alta (es. 0.7 per essere conservativi)

custom_threshold = 0.8
y_pred_holdout = (y_probs_holdout >= custom_threshold).astype(int)


# Calcola la matrice di confusione
cm = confusion_matrix(y_holdout, y_pred_holdout)

#-----------------------------------------------------------------------------#
#    PER IL GRAFICO DELLA MATRICE DI CONFUSIONE ANDARE IN FONDO AL CODICE     #
#-----------------------------------------------------------------------------#

# Report dettagliato per controllare la Recall, Precision e f1-score

print("\n--- PERFORMANCE REPORT ---")
print(classification_report(y_holdout, y_pred_holdout))


# 1. Calcolo dei punti della curva ROC
# y_probs_holdout sono le probabilità della classe "SI" (1) calcolate in precedenza

fpr, tpr, thresholds = roc_curve(y_holdout, y_probs_holdout)

# 2. Calcolo dell'area sotto la curva (AUC)
roc_auc = auc(fpr, tpr)

#-----------------------------------------------------------------------------#
#         PER IL PLOT DELLA CURVA AUC ANDARE IN FONDO AL PLOT                 #
#-----------------------------------------------------------------------------#


###############################################################################
#               CONFRONTO TRA ESITO FINALE E SCORE SONEPAR                    #
###############################################################################

# Recuperiamo gli indici dell'Holdout per estrarre i dati originali dal df iniziale

holdout_indices = X_holdout.index

# Creiamo il df di confronto tra esito finale e lo score sonepar per valutare le incoerenze

confronto_df = pd.DataFrame({
    'ID': X_holdout.index,
    'Esito_Reale': y_holdout.values.flatten(),
    'Predizione_Modello': y_pred_holdout,
    'Prob_Modello_SI': y_probs_holdout,
    'Score_Sonepar': df.loc[X_holdout.index, 'SCORE_SONEPAR'].values,
    'ITP_Reale': df.loc[X_holdout.index, 'ITP'].values,
    'Anzianita_Giorni': df_prep.loc[X_holdout.index, 'ETA_IMPRESA_GIORNI'].values
})

confronto_df['Rating_Numerico'] = confronto_df['Score_Sonepar'].apply(mappa_sonepar)

# Normalizziamo il punteggio Sonepar su scala 0-1 per renderlo confrontabile con Prob_Modello_SI
# 7.3 è il punteggio massimo teorico (1 + 0.3), 0.4 il minimo (1 - 0.6)

# LO SCORE SONEPAR01 è UN PROXY DELLA PROBABILITÀ CHE SONEAPR ASSEGNA TRAMITE IL SUO SCORE
# DI CONCEDERE IL FIDO (NB: AVVOLTE QUANDO LO SCORE è BASSO SONEPAR CONCEDE COMUNQUE IL FIDO)

min_s, max_s = 0.4, 7.3
confronto_df['Sonepar_01'] = (confronto_df['Rating_Numerico'] - min_s) / (max_s - min_s)

# Calcolo del Disaccordo Assoluto (È LA DIFFERENZA TRA LA PROBABILITÀ CHE IL NOSTRO MODELLO
# ASSEGNA ALLA CONCESSIONE DEL FIDO E QUELLA ASSEGNATA DALLO SCORE. TANTO PIÙ è ALTA TANTO
# PIÙ DISACCORDO C'È TRA I DUE MODELLI)

confronto_df['Disaccordo_Assoluto'] = np.abs(confronto_df['Sonepar_01'] - confronto_df['Prob_Modello_SI'])

# FILTRO TOP 10 INCOERENZE 
# Filtriamo: Esito diverso dalla predizione E Score non nullo

incoerenze_top10 = confronto_df[
    (confronto_df['Esito_Reale'] != confronto_df['Predizione_Modello']) & 
    (confronto_df['Score_Sonepar'] != 'ND')
].sort_values(by='Disaccordo_Assoluto', ascending=False).head(10)

print("--- TOP 10 INCOERENZE: MODELLO VS RATING ALFANUMERICO SONEPAR ---")

# Usiamo i nomi corretti delle colonne per il print

print(incoerenze_top10[['ID', 'Esito_Reale', 'Score_Sonepar', 'Prob_Modello_SI', 'ITP_Reale', 'Disaccordo_Assoluto']])

#------------------------------------------------------------------------------------------#
# PER I GRAFICI DI CONFRONTO TRA SCORE SONEPAR E NOSTRO MODELLO ANDARE IN FONDO AL CODICE  #
#------------------------------------------------------------------------------------------#

###############################################################################
#  AGGIUNTA DI REGOLE SULLA BASE DELLO STATO ATTIVITA E NATURA GIURIDICA      #
###############################################################################

## Prepariamo il dataframe di confronto ad accogliere le regole aggiuntive del
## motore decisonale di sonepar che sono lo stato attività e natura giuridica.

confronto_df = confronto_df.merge(
    df_prep[['SINTESI_NATURA_GIURIDICA']],
    left_on='ID', 
    right_index=True, 
    how='left'
)


confronto_df['Verdetto_Finale'] = confronto_df.apply(motore_decisionale, axis=1)

# Creiamo una colonna che raggruppa il verdetto per il confronto
confronto_df['Verdetto_Sintetico'] = np.where(
    confronto_df['Verdetto_Finale'] == 'APPROVATO', 1, 0
)

###############################################################################
#  CONFRONTO TRA MATRICI DI CONFUSIONE PRIMA E DOPO L'AGGIUNTA DELLE REGOLE   #
###############################################################################

# Creiamo una copia per non sporcare il dataframe originale
df_plot = confronto_df.copy()

# Trasformiamo il Verdetto_Finale in 0 e 1 (SI=1, tutto il resto=0)
# Usiamo .str.contains per gestire le varie stringhe di RIFIUTATO
df_plot['Esito_Aggiustato'] = df_plot['Verdetto_Finale'].apply(lambda x: 1 if x == 'APPROVATO' else 0)

# Assicuriamoci che Esito_Reale e Predizione_Modello siano interi
df_plot['Esito_Reale'] = df_plot['Esito_Reale'].astype(int)
df_plot['Predizione_Modello'] = df_plot['Predizione_Modello'].astype(int)

# --- 2. RIMOZIONE NA ---
# Rimuoviamo eventuali righe che hanno NaN in una di queste 3 colonne
df_plot = df_plot.dropna(subset=['Esito_Reale', 'Predizione_Modello', 'Esito_Aggiustato'])

# --- 3. GENERAZIONE MATRICI ---
y_true = df_plot['Esito_Reale']
y_pred_base = df_plot['Predizione_Modello']
y_pred_final = df_plot['Esito_Aggiustato']

cm_base = confusion_matrix(y_true, y_pred_base)
cm_final = confusion_matrix(y_true, y_pred_final)

#-----------------------------------------------------------------------------#
#PER GRAFICO DI CONFRONTO DELLE MATRICI DI CONFUSIONE ANDARE IN FONDO AL CODICE#
#-----------------------------------------------------------------------------#

###############################################################################
#                              GRAFICI                                        #
###############################################################################

#-----------------------------------------------------------------------------#
#    ISTOGRAMMI PER OSSERVARE LA DISTRBUZIONE DELLE VARIABILI                 #
#-----------------------------------------------------------------------------#

## Istogramma PRIMA della trasformazione logaritmica di età impresa

plt.figure(figsize=(10,6))
sns.histplot(df_prep["ETA_IMPRESA_GIORNI"], bins=50, kde=True, color="skyblue")
plt.title("Distribuzione ETA_IMPRESA_GIORNI")
plt.xlabel("ETA_IMPRESA_GIORNI")
plt.ylabel("Frequenza")
plt.show()

## Istogramma DOPO della trasformazione logaritmica dI età impresa

plt.figure(figsize=(10,6))
sns.histplot(df_prep["LOG_ETA_IMPRESA"], bins=50, kde=True, color="skyblue")
plt.title("Distribuzione ETA_IMPRESA_GIORNI")
plt.xlabel("ETA_IMPRESA_GIORNI")
plt.ylabel("Frequenza")
plt.show()

## Istogramma PRIMA della trasformazione logaritmica di ITP

plt.figure(figsize=(10,6))
sns.histplot(df_prep['ITP'], bins=50, kde=True, color="lightcoral")
plt.title("Distribuzione ITP")
plt.xlabel("ITP")
plt.ylabel("Frequenza")
plt.show()


## Istogramma DOPO della trasformazione logaritmica di ITP
plt.figure(figsize=(10,6))
sns.histplot(df_prep['LOG_ITP'], bins=50, kde=True, color="lightcoral")
plt.title("Distribuzione ITP")
plt.xlabel("ITP")
plt.ylabel("Frequenza")
plt.show()

## Istogramma PRIMA della trasformazione logaritmica di DIPENDENTI
plt.figure(figsize=(10,6))
sns.histplot(df_prep['DIPENDENTI'], bins=50, kde=True, color="lightcoral")
plt.title("Distribuzione DIPENDENTI")
plt.xlabel("Dipendenti")
plt.ylabel("Frequenza")
plt.show()

## Istogramma DOPO della trasformazione logaritmica di DIPENDENTI
plt.figure(figsize=(10,6))
sns.histplot(df_prep['LOG_DIPENDENTI'], bins=50, kde=True, color="lightcoral")
plt.title("Distribuzione LOG_DIPENDENTI")
plt.xlabel("Dipendenti(log)")
plt.ylabel("Frequenza")
plt.show()

## Istogramma PRIMA della trasformazione logaritmica di NUMERO IMMOBILI
plt.figure(figsize=(10,6))
sns.histplot(df_prep['NUMERO_IMMOBILI'], bins=50, kde=True, color="lightcoral")
plt.title("Distribuzione NUMERO_IMMOBILI")
plt.xlabel("Numero immobili")
plt.ylabel("Frequenza")
plt.show()

## Istogramma DOPO della trasformazione logaritmica di NUMERO IMMOBILI

plt.figure(figsize=(10,6))
sns.histplot(df_prep['LOG_NUMERO_IMMOBILI'], bins=50, kde=True, color="lightcoral")
plt.title("Distribuzione LOG_NUMERO_IMMOBILI")
plt.xlabel("Numero immobili(log)")
plt.ylabel("Frequenza")
plt.show()

#-----------------------------------------------------------------------------#
#                        LEARNING CURVES DEL MODELLO                          #
#-----------------------------------------------------------------------------#

# Creazione del grafico
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

plt.title("Learning Curves (Logistic Regression)")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid(True)
plt.show()

#-----------------------------------------------------------------------------#
#        MATRICE DI CONFUSIONE 1° FASE DELLA CREAZIONE DEL MODELLO            #
#-----------------------------------------------------------------------------#

disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['NO (Rifiutato)', 'SI (Concesso)'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice di Confusione - Holdout Set (Soglia 0.8)")
plt.show()

#-----------------------------------------------------------------------------#
#                            CURVA ROC E AUC                                  #
#-----------------------------------------------------------------------------#

# 3. Creazione del grafico
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificatore Casuale (Area = 0.5)')

# Estetica del grafico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di Falsi Positivi (FPR)', fontsize=12)
plt.ylabel('Tasso di Veri Positivi (TPR / Recall)', fontsize=12)
plt.title('Curva ROC - Capacità Predittiva del Modello AI', fontsize=15)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

#-----------------------------------------------------------------------------#
#        SCATTER PLOT: PROBABILITÀ MODELLO VS RATING SONEPAR                  #
#-----------------------------------------------------------------------------#

# --- PREPARAZIONE DATI PER I GRAFICI ---
# Definiamo una categoria di "Accordo" per semplificare la visualizzazione
# Se il disaccordo è basso (< 0.25), i due sistemi sono allineati
confronto_df['Stato_Accordo'] = np.where(confronto_df['Disaccordo_Assoluto'] < 0.25, 'Accordo', 'Disaccordo')

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=confronto_df, 
    x='Prob_Modello_SI', 
    y='Sonepar_01', 
    hue='Stato_Accordo',
    palette={'Accordo': '#2ecc71', 'Disaccordo': '#e74c3c'},
    alpha=0.7,
    s=100
)

# Linea di accordo perfetto
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Accordo Perfetto')

plt.title('Discrepanza tra Modello AI e Rating Sonepar', fontsize=15)
plt.xlabel('Probabilità di Successo (Modello)', fontsize=12)
plt.ylabel('Affidabilità Decodificata (Sonepar)', fontsize=12)
plt.legend(title="Stato")
plt.grid(True, alpha=0.3)
plt.show()

#-----------------------------------------------------------------------------#
#       BOXPLOT: PERCHÉ IL MODELLO LITIGA CON SONEPAR? (FOCUS SU ITP)         #
#-----------------------------------------------------------------------------#

plt.figure(figsize=(10, 6))

# Confrontiamo l'ITP (ritardo pagamenti) nei casi di accordo e disaccordo

sns.boxplot(
    data=confronto_df, 
    x='Stato_Accordo', 
    y='ITP_Reale', 
    palette='Set2'
)

plt.title('Impatto dell\'ITP sulle Incoerenze di Rating', fontsize=15)
plt.xlabel('Relazione tra Modello e Sonepar', fontsize=12)
plt.ylabel('ITP (Giorni di ritardo)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()

#-----------------------------------------------------------------------------#
#          DISTRIBUZIONE DELLE PROBABILITÀ PER CLASSE SONEPAR                 #
#-----------------------------------------------------------------------------#


# Raggruppiamo i rating Sonepar in macro-classi per chiarezza
def raggruppa_sonepar(score):
    if str(score).startswith(('1', '2')): return 'Rischio Basso (1-2)'
    if str(score).startswith(('3', '4')): return 'Rischio Medio (3-4)'
    if str(score).startswith(('5', '6', '7')): return 'Rischio Alto (5-7)'
    return 'Altro/ND'

confronto_df['Classe_Sonepar'] = confronto_df['Score_Sonepar'].apply(raggruppa_sonepar)

plt.figure(figsize=(12, 6))
sns.violinplot(
    data=confronto_df[confronto_df['Classe_Sonepar'] != 'Altro/ND'], 
    x='Classe_Sonepar', 
    y='Prob_Modello_SI', 
    order=['Rischio Basso (1-2)', 'Rischio Medio (3-4)', 'Rischio Alto (5-7)'],
    inner="quartile",
    palette='viridis'
)

plt.title('Distribuzione Probabilità Modello per Classe di Rischio Sonepar', fontsize=15)
plt.xlabel('Classe di Rischio Sonepar', fontsize=12)
plt.ylabel('Probabilità di Successo (Modello)', fontsize=12)
plt.savefig('violin_distribuzione.png')
plt.show()

#-----------------------------------------------------------------------------#
#          MATRICI DI CONFUSIONE PRE E POST AGGIUNTA DELLE REGOLE             #
#-----------------------------------------------------------------------------#


# --- 1. SETUP COLORI E GRADIENTI ---
abisso = '#0A1F1A'
smeraldo = '#0F5040'
particella = '#00CFA8'
nebbia = '#C2FFF3' # Usiamo la Nebbia come base chiara per lo "zero"

# Creiamo due mappe di colore (Light -> Dark)
cmap_smeraldo = LinearSegmentedColormap.from_list("Smeraldo_Grad", [nebbia, smeraldo])
cmap_particella = LinearSegmentedColormap.from_list("Particella_Grad", [nebbia, particella])

labels = ['Non Concesso', 'Concesso'] # Basato su SI/NO del business case 

# --- 2. VISUALIZZAZIONE ---
fig, ax = plt.subplots(1, 2, figsize=(22, 10))

# Font giganti per la massima leggibilità
font_numeri = {'size': 30, 'weight': 'bold'} 
font_titoli = {'size': 24, 'weight': 'bold', 'color': abisso}
font_etichette = {'size': 30, 'weight': 'bold'}

# --- 2. VISUALIZZAZIONE ---
fig, ax = plt.subplots(1, 2, figsize=(22, 10))

# Definiamo una dimensione per i tick labels (Concesso/Non Concesso)
font_tick_labels = {'size': 22, 'weight': 'bold'} 

# --- MATRICE 1: SOLO MODELLO AI ---
sns.heatmap(cm_base, annot=True, fmt='d', cmap=cmap_smeraldo, ax=ax[0],
            xticklabels=labels, yticklabels=labels, 
            annot_kws=font_numeri, cbar=False)

ax[0].set_title('1. SOLO MODELLO', fontdict=font_titoli, pad=35)
ax[0].set_xlabel('ESITO PREDETTO', fontdict=font_etichette, labelpad=20)
ax[0].set_ylabel('ESITO REALE', fontdict=font_etichette, labelpad=20)

# AGGIUNTA: Ingrandiamo 'Concesso' e 'Non Concesso'
ax[0].tick_params(axis='both', which='major', labelsize=25) # Imposta la dimensione per entrambi gli assi
# Opzionale: Ruotiamo le etichette sull'asse Y per renderle orizzontali e più leggibili
ax[0].set_yticklabels(labels, va='center', rotation=0)

# --- MATRICE 2: AI + REGOLE BUSINESS ---
sns.heatmap(cm_final, annot=True, fmt='d', cmap=cmap_particella, ax=ax[1],
            xticklabels=labels, yticklabels=labels, 
            annot_kws=font_numeri, cbar=False)

ax[1].set_title('2. MODELLO + REGOLE BUSINESS\n(Filtri Stato Attività)', fontdict=font_titoli, pad=35)
ax[1].set_xlabel('ESITO PREDETTO', fontdict=font_etichette, labelpad=20)
ax[1].set_ylabel('ESITO REALE', fontdict=font_etichette, labelpad=20)

# AGGIUNTA: Ingrandiamo 'Concesso' e 'Non Concesso' anche qui
ax[1].tick_params(axis='both', which='major', labelsize=25)
ax[1].set_yticklabels(labels, va='center', rotation=0)

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------#
#                GRAFICO FEATURE IMPORTANCE (Coefficienti)                    #
#-----------------------------------------------------------------------------#

plt.figure(figsize=(10, 6))

# Ordiniamo per valore del coefficiente per una lettura più chiara

importance_plot = importance_df.sort_values(by='Coefficiente (Peso)', ascending=False)

sns.barplot(
    data=importance_plot, 
    x='Coefficiente (Peso)', 
    y='Variabile', 
    palette='vlag' # Palette divergente: evidenzia la differenza tra impatti positivi e negativi
)

plt.title('Peso delle Variabili sulla Decisione di Fido (Modello Ridge)', fontsize=14)
plt.xlabel('Coefficiente (Impatto sulla Probabilità)', fontsize=12)
plt.ylabel('Variabile Analizzata', fontsize=12)
plt.axvline(0, color='black', lw=1) # Linea centrale sullo zero per separare i fattori
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('1_feature_importance.png')
plt.show()

#-----------------------------------------------------------------------------#
#     BOXPLOT APPROVAZIONE DEI FIDI PER REVISONE = SI E REVISONE = NO         #
#-----------------------------------------------------------------------------#
if 'REVISIONE' not in confronto_df.columns:
    confronto_df = confronto_df.merge(df_prep[['REVISIONE']], left_on='ID', right_index=True, how='left')

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=confronto_df, 
    x='REVISIONE', 
    y='Prob_Modello_SI', 
    palette=['#0D3B2E','#4DFFD8']
)

plt.xticks([0, 1], ['Nuovo Cliente (0)', 'Cliente Storico (1)'])
plt.title('Impatto della Relazione Pregressa sulla Probabilità di Approvazione', fontsize=14)
plt.xlabel('Stato Revisione (Fiducia Storica)', fontsize=12)
plt.ylabel('Probabilità di Successo stimata dal Modello', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('2_prob_vs_revisione.png')
plt.show()

#-----------------------------------------------------------------------------#
#            ITP VS PROBABILITÀ (Analisi del Rischio di Ritardo)              #
#-----------------------------------------------------------------------------#


plt.figure(figsize=(10, 6))

# Scatter plot con linea di trend per evidenziare la correlazione negativa
sns.regplot(
    data=confronto_df, 
    x='ITP_Reale', 
    y='Prob_Modello_SI', 
    scatter_kws={'alpha':0.4, 's':30, 'color':'teal'},
    line_kws={'color':'red', 'label':'Trend di Decadimento'}
)

plt.title('ITP come Segnale di Allerta: Impatto dei Ritardi sulla Probabilità', fontsize=14)
plt.xlabel('Indice di Tempestività di Pagamento (Giorni di ritardo medi)', fontsize=12)
plt.ylabel('Probabilità di Approvazione', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3_itp_vs_probabilita.png')
plt.show()

#-----------------------------------------------------------------------------#
#        GRAFICO TASSO DI APPROVAZIONE VS MASSA MONETARIA PER FASCIA          #
#-----------------------------------------------------------------------------#

# --- 1. SETUP PALETTE E DATI ---
abisso = '#0A1F1A'
foresta = '#0D3B2E'
smeraldo = '#0F5040'
particella = '#00CFA8'
acquamarina = '#4DFFD8'
nebbia = '#C2FFF3'

df.columns = df.columns.str.strip()
COL_ESITO = 'Esito_finale' 
COL_FASCIA = 'Fascia Fido'
COL_VALORE = 'VALORE FIDO RICHIESTO'

if df[COL_VALORE].dtype == 'object':
    df[COL_VALORE] = pd.to_numeric(df[COL_VALORE].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

fascia_order = ['Fino a 5000', 'Tra 5000 e 10000', 'Tra 10000 e 20000']

summary = df.groupby(COL_FASCIA).agg(
    Total_Count=(COL_ESITO, 'count'),
    Approved_Count=(COL_ESITO, lambda x: (x == 'SI').sum()),
    Total_Value=(COL_VALORE, 'sum')
).reindex(fascia_order)

summary['Approval_Rate'] = (summary['Approved_Count'] / summary['Total_Count']) * 100
summary['Rejection_Rate'] = 100 - summary['Approval_Rate']
summary['Total_Value_K'] = summary['Total_Value'] / 1000

# --- 2. GRAFICO CON LEGENDA ESPLICITA ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 14), sharey=True)

# Grafico 1: VOLUME
summary[['Rejection_Rate', 'Approval_Rate']].plot(
    kind='barh', stacked=True, ax=ax1, color=[abisso, particella], alpha=0.9, width=0.7
)

ax1.set_title('1. PERFORMANCE PER VOLUME\n(% Numero Pratiche)', fontsize=28, fontweight='bold', pad=40, color=abisso)
ax1.set_xlabel('Percentuale (%)', fontsize=22, labelpad=20)
ax1.set_ylabel('FASCIA FIDO', fontsize=26, fontweight='bold', labelpad=25)
ax1.tick_params(axis='both', labelsize=20)

# --- CREAZIONE LEGENDA PERSONALIZZATA ---
# Creiamo dei "quadratini" colorati per la legenda
no_patch = mpatches.Patch(color=abisso, label='Non Concesso (NO)')
si_patch = mpatches.Patch(color=particella, label='Concesso (SI)')

# La posizioniamo sotto l'asse X del primo grafico
ax1.legend(handles=[no_patch, si_patch], 
           fontsize=22, 
           loc='upper center', 
           bbox_to_anchor=(0.5, -0.15), 
           ncol=2, 
           frameon=False)

for i, (rej, app) in enumerate(zip(summary['Rejection_Rate'], summary['Approval_Rate'])):
    ax1.text(rej/2, i, f'{rej:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=22)
    ax1.text(rej + app/2, i, f'{app:.1f}%', va='center', ha='center', color=abisso, fontweight='bold', fontsize=22)

# Grafico 2: MASSA
sns.barplot(x='Total_Value_K', y=summary.index, data=summary, color=smeraldo, ax=ax2, alpha=0.9, width=0.7)

ax2.set_title('2. MASSA MONETARIA RICHIESTA\n(Valore Totale in €)', fontsize=28, fontweight='bold', pad=40, color=abisso)
ax2.set_xlabel('Migliaia di Euro (k€)', fontsize=22, labelpad=20)
ax2.tick_params(axis='x', labelsize=20)

for i, v in enumerate(summary['Total_Value']):
    valore_formattato = f"€ {int(v):,}".replace(",", ".")
    ax2.text(summary['Total_Value_K'].iloc[i] + 5, i, 
             f"{valore_formattato}\n({summary['Total_Count'].iloc[i]} pratiche)", 
             va='center', fontweight='bold', fontsize=22, color=foresta)

# REGOLAZIONE FINALE PER NON TAGLIARE LA LEGENDA
plt.tight_layout()
plt.subplots_adjust(bottom=0.2) # Questo comando crea lo spazio bianco sotto per la legenda

plt.show()