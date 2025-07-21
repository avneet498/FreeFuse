#!/usr/bin/env python
# coding: utf-8

# LOADING DATASET

# In[ ]:


import pandas as pd

# Define the file path
file_path = r"C:\Users\carlo\OneDrive - Langara College\Langara College - Courses\DANA 4850 Projects\FreeFuse\FreeFuse Sentiment Emotion Analysis [Langara].xlsx"

# Load the Excel file
excel_file = pd.ExcelFile(file_path)
sheet_names = excel_file.sheet_names

# Dictionary to store columns from each sheet
sheet_columns = {}

# Read columns for each sheet (headers only)
for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet, nrows=0)
    sheet_columns[sheet] = set(df.columns)


# CHECKING COLUMNS NAMES

# In[ ]:


# Choose the first sheet as reference
reference_sheet = list(sheet_columns.keys())[0]
reference_columns = sheet_columns[reference_sheet]

print(f"Reference Sheet: {reference_sheet}")
print(f"Reference Columns: {reference_columns}")
print("\nComparison:\n" + "-"*50)

# Compare columns in other sheets
for sheet, columns in sheet_columns.items():
    if sheet == reference_sheet:
        continue

    if reference_columns == columns:
        print(f"\nSheet: {sheet} --> Columns MATCH exactly.")
    else:
        print(f"\nSheet: {sheet} --> Columns DO NOT match.")


# MERGING DATASETS

# In[ ]:


# Read all sheets fully into a list of DataFrames
dfs = [pd.read_excel(excel_file, sheet_name=sheet) for sheet in sheet_names]

# Merge (concatenate) all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)

# Show result
print(f"Merged dataset shape: {merged_df.shape}")
print(merged_df.head())


# SORTING THE "CONTENTID" IN ALPHABETICAL ORDER

# In[ ]:


from tabulate import tabulate

# Sort merged dataframe by 'ContentID'
merged_df = merged_df.sort_values(by='ContentID')

# Print first 20 rows in formatted table
print(tabulate(merged_df.head(20), headers='keys', tablefmt='fancy_grid'))


# COUNTING THE NUMBER OF DIFFERENT NUMBER OF CONTENTS AND USERS

# In[ ]:


# Count unique ContentID values
unique_content_ids = merged_df['ContentID'].nunique()
print(f"THERE ARE {unique_content_ids} DIFFERENT ContentID IN THIS DATASET.")

# Count unique UserID values
unique_user_ids = merged_df['UserID'].nunique()
print(f"THERE ARE {unique_user_ids} DIFFERENT UserID IN THIS DATASET.")

# Get total number of rows
num_rows = len(merged_df)
print(f"THE DATASET HAS {num_rows} ROWS.")


# INDEXING THE USER AND CONTENT ID
# 

# In[ ]:


# Create mappings for UserID and ContentID
user_id_map = {old_id: f"User{str(i+1).zfill(2)}" for i, old_id in enumerate(merged_df['UserID'].unique())}
content_id_map = {old_id: f"Content{str(i+1).zfill(2)}" for i, old_id in enumerate(merged_df['ContentID'].unique())}

# Apply the mappings to the merged DataFrame
merged_df['UserID'] = merged_df['UserID'].map(user_id_map)
merged_df['ContentID'] = merged_df['ContentID'].map(content_id_map)


# DATA ENTRY ERROS CHECK (UserId, ContentID, Node Title, Sentiment, Emotion)

# In[ ]:


from tabulate import tabulate

# Loop through all columns in merged_df
for col in merged_df.columns:
    # Count occurrences of each unique value (drop NaN if desired)
    value_counts = merged_df[col].value_counts(dropna=False)

    # Prepare data for tabulate
    table_data = list(zip(value_counts.index.astype(str), value_counts.values))

    print(f"\nClass Distribution - {col}")
    print(tabulate(table_data, headers=[col, 'Count'], tablefmt='fancy_grid'))


# IT WAS IDENTIFIED DATA ENTRY ERROR IN SOME VARIABLES. THE "Excitement" should be in Emotion (Class Distribution-Emotion chart shows it), not in "Sentiment". THE "Picked up some great insights." should be in comments (Class Distribution - Node Tittle shows it) not in "Node Title". AFTER THIS ANALYSIS, IT IS POSSIBLE TO SEE THAT THE DATA ENTRY CLERK FORGOT TO FILL THE NODE TITLE CORRECTLY, RESULT IN A NAN VALUE IN EMOTION, THIS ROW WILL BE ELIMINATED

# In[ ]:


# Remove rows where Sentiment is 'Excitement'
merged_df = merged_df[merged_df['Sentiment'] != 'Excitement']


# CHECKING FOR THE EXISTENCE OF MISSING VALUES

# In[ ]:


# Check for missing values in each column for merged_df
missing_values = merged_df.isnull().sum()

# Display the result
print("Missing values per column:")
print(missing_values[missing_values > 0])


# CHECKING FOR THE EXISTENCE OF DUPLICATED ROWS

# In[ ]:


from tabulate import tabulate

# Find duplicated rows in merged_df (keeping first occurrence to count total duplicates)
duplicate_rows = merged_df[merged_df.duplicated()]

# Count total number of duplicate rows
total_duplicates = duplicate_rows.shape[0]

# Sort duplicates for better visualization
duplicate_rows_sorted = duplicate_rows.sort_values(by=['UserID', 'ContentID', 'Node Title'])

# Display total number of duplicate rows
print(f"Total duplicated rows (excluding first occurrences): {total_duplicates}")

# Display duplicates in a formatted table
if total_duplicates > 0:
    print(tabulate(duplicate_rows_sorted, headers='keys', tablefmt='fancy_grid', showindex=False))
else:
    print("No duplicates found.")


# HANDLING DUPLICATED ROWS

# In[ ]:


# Remove duplicated rows from merged_df, keeping the first occurrence
merged_df_cleaned = merged_df.drop_duplicates(keep='first')

# Check the new shape and confirm duplicates are removed
print(f"Original shape: {merged_df.shape}")
print(f"New shape after removing duplicates: {merged_df_cleaned.shape}")


# CHECK ENGAGEMENT RATE VALID RANGE

# In[ ]:


from tabulate import tabulate

# Ensure 'EngagementRate(%)' is numeric in merged_df
merged_df['EngagementRate(%)'] = pd.to_numeric(merged_df['EngagementRate(%)'], errors='coerce')

# Filter rows where Engagement Rate is outside the 0–100 range
out_of_range = merged_df[(merged_df['EngagementRate(%)'] < 0) | (merged_df['EngagementRate(%)'] > 100)]

# Display result
print(f"❗ Number of rows with EngagementRate(%) outside the 0–100 range: {len(out_of_range)}")

# Show rows if any exist
if not out_of_range.empty:
    print(tabulate(out_of_range[['UserID', 'ContentID', 'EngagementRate(%)']], headers='keys', tablefmt='fancy_grid', showindex=False))


# THE ONLY NUMERICAL VARIABLE IN THIS DATASET IS ENGAGEMENT RATE, BUT THIS VARIABLE SHOULD NOT BE TREATED AS HAVING OUTLIERS BECAUSE IT REFLECTS THE PERSONAL PERSPECTIVE OF EACH VIEWER. ADDITIONALLY, ALL VALUES FALL WITHIN THE VALID RANGE OF 0% TO 100%. THEREFORE, THERE IS NO NEED TO HANDLE OUTLIERS.

# CLASS BALANCE 1
# 

# In[ ]:


import pandas as pd

# Ensure no missing values in classification columns
sentiment_counts = merged_df['Sentiment'].dropna().value_counts(normalize=True).rename_axis('Sentiment').reset_index(name='Proportion')
emotion_counts = merged_df['Emotion'].dropna().value_counts(normalize=True).rename_axis('Emotion').reset_index(name='Proportion')

# Display
print("📊 Sentiment Class Distribution:")
print(sentiment_counts)

print("\n💖 Emotion Class Distribution:")
print(emotion_counts)


# TEXT + EMOJIS + ROBERTA + SMOTE
# 

# In[ ]:


# 1. Imports
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, AutoModel
import emoji
from collections import Counter

# 2. Combine comment and reaction (convert emojis to text)
def preprocess_row(row):
    comment = str(row['Comment'])
    reaction = emoji.demojize(str(row['Reaction']), delimiters=(" ", " "))
    return f"{comment} {reaction}"

merged_df_cleaned['input_text'] = merged_df_cleaned.apply(preprocess_row, axis=1)

# 3. Load RoBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model.eval()  # Set to inference mode

# 4. Extract sentence embeddings (CLS token)
def get_roberta_embedding(text):
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        output = model(**encoded)
        return output.last_hidden_state[0, 0, :].numpy()

# 5. Generate embeddings for all input texts
embeddings = []
for text in tqdm(merged_df_cleaned['input_text'], desc="Extracting embeddings"):
    emb = get_roberta_embedding(text)
    embeddings.append(emb)

embeddings_matrix = np.vstack(embeddings)

# 6. Ensure 'EngagementRate(%)' is numeric
merged_df_cleaned['EngagementRate(%)'] = pd.to_numeric(merged_df_cleaned['EngagementRate(%)'], errors='coerce')

# 7. Encode the target variable (e.g., Sentiment or Emotion)
label_encoder = LabelEncoder()
merged_df_cleaned['target'] = label_encoder.fit_transform(merged_df_cleaned['Sentiment'])  # or 'Emotion'

# 8. Normalize engagement rate
scaler = MinMaxScaler()
merged_df_cleaned['engagement_scaled'] = scaler.fit_transform(merged_df_cleaned[['EngagementRate(%)']])

# 9. Build the feature matrix (embeddings + engagement)
X = np.hstack((embeddings_matrix, merged_df_cleaned[['engagement_scaled']].values))
y = merged_df_cleaned['target']

# 10. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 11. Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 12. Show the new class distribution
print("📊 Class distribution after SMOTE:", Counter)


# CLASS BALANCE 2

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# 1. Class distribution before SMOTE
original_counts = Counter(y_train)
df_original = pd.DataFrame.from_dict(original_counts, orient='index').reset_index()
df_original.columns = ['Class', 'Count']
df_original['Class'] = df_original['Class'].map(lambda x: label_encoder.inverse_transform([x])[0])

# 2. Class distribution after SMOTE
balanced_counts = Counter(y_train_bal)
df_balanced = pd.DataFrame.from_dict(balanced_counts, orient='index').reset_index()
df_balanced.columns = ['Class', 'Count']
df_balanced['Class'] = df_balanced['Class'].map(lambda x: label_encoder.inverse_transform([x])[0])

# 3. Side-by-side bar plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=df_original, x='Class', y='Count', ax=axes[0])
axes[0].set_title('Class Distribution Before SMOTE')
axes[0].set_ylabel("Count")

sns.barplot(data=df_balanced, x='Class', y='Count', ax=axes[1])
axes[1].set_title('Class Distribution After SMOTE')
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


# XGBoost Optuna tunning

# In[ ]:


import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "mlogloss"
    }

    model = XGBClassifier(**params)
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro') 

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=30)

print("🔧 Best XGBoost params:", study_xgb.best_params)


# LightGBM Optuna tunning

# In[ ]:


from lightgbm import LGBMClassifier

def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42
    }

    model = LGBMClassifier(**params)
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')

study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=30)

print("🔧 Best LightGBM params:", study_lgb.best_params)


# Random Forest Optuna tunning

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.metrics import f1_score

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')  # macro = considera todas as classes igualmente

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=30)

print("🔧 Best Random Forest params:", study_rf.best_params)


# TRAINNING THE TUNNED MODELS

# In[ ]:


from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Recriar modelos com os melhores parâmetros
best_rf = RandomForestClassifier(
    n_estimators=191, max_depth=14,
    min_samples_split=5, min_samples_leaf=2,
    max_features=None, class_weight='balanced', random_state=42
)

best_xgb = XGBClassifier(
    n_estimators=315, max_depth=4, learning_rate=0.1809,
    subsample=0.9999, colsample_bytree=0.5874, gamma=3.632,
    use_label_encoder=False, eval_metric="mlogloss", random_state=42
)

best_lgb = LGBMClassifier(
    n_estimators=227, max_depth=9, learning_rate=0.0218,
    num_leaves=122, subsample=0.7734, colsample_bytree=0.5522,
    random_state=42
)

# Treinar e avaliar
models = {"Random Forest": best_rf, "XGBoost": best_xgb, "LightGBM": best_lgb}
f1_scores = {}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)

    print(f"\n📌 {name} – Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    f1_scores[name] = f1_score(y_test, y_pred, average=None)

# Plot F1 por classe
labels = label_encoder.classes_
x = range(len(labels))

plt.figure(figsize=(10, 6))
for name, scores in f1_scores.items():
    plt.plot(x, scores, marker='o', label=name)

plt.xticks(x, labels)
plt.title("F1-Score por classe – Comparativo de Modelos Tunados")
plt.ylabel("F1-Score")
plt.xlabel("Classe")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# SAVING BEST MODEL

# In[ ]:


# 1. Imports
import joblib
from lightgbm import LGBMClassifier

# 2. Best parameters from Optuna
best_params = {
    'n_estimators': 227,
    'max_depth': 9,
    'learning_rate': 0.021811832317250604,
    'num_leaves': 122,
    'subsample': 0.7734026280620399,
    'colsample_bytree': 0.5522628090905064
}

# 3. Train the best model
best_model = LGBMClassifier(**best_params, random_state=42)
best_model.fit(X_train_bal, y_train_bal)

# 4. Save the model to .pkl
joblib.dump(best_model, 'lightgbm_best_model.pkl')
print("✅ Model saved to 'lightgbm_best_model.pkl'")


# SENTIMENT CLASSIFICATION MODEL

# CHATBOT

# In[1]:


import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import emoji
from sklearn.preprocessing import LabelEncoder
import warnings

# Silenciar warnings do sklearn/lightgbm
warnings.filterwarnings("ignore")

# === Load model and tokenizer ===
model = joblib.load("lightgbm_best_model.pkl")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta.eval()

# === Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Negative', 'Neutral', 'Positive'])

# === Emoji map
emoji_map = {
    1: "😀", 2: "😃", 3: "😄", 4: "😁", 5: "😆", 6: "😅", 7: "😂",
    8: "🤣", 9: "❤️", 10: "😍", 11: "🥰", 12: "😐", 13: "😕", 14: "😟",
    15: "🤔", 16: "🙁", 17: "☹️", 18: "😞", 19: "😢", 20: "😭", 21: "😡",
    22: "😠", 23: "🤯", 24: "💔", 25: "🥺", 26: "😨", 27: "😳"
}

def show_emoji_options():
    print("\n🎭 Emoji Options:")
    for i in sorted(emoji_map.keys()):
        print(f"{i}: {emoji_map[i]}", end="  ")
    print("\n")

def get_roberta_embedding(text):
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        output = roberta(**encoded)
        return output.last_hidden_state[0, 0, :].numpy()

# === Chatbot start ===
print("🤖 Sentiment Chatbot is running!")
show_emoji_options()

# --- Get user inputs
comment = input("\n💬 Enter a comment: ").strip()
emoji_numbers = input(
    "😄 Enter emoji number(s) separated by space (e.g. 1 9 21):\n"
    "1: 😀  2: 😃  3: 😄  4: 😁  5: 😆  6: 😅  7: 😂  8: 🤣  9: ❤️  10: 😍  11: 🥰\n"
    "12: 😐  13: 😕  14: 😟  15: 🤔  16: 🙁  17: ☹️  18: 😞  19: 😢  20: 😭\n"
    "21: 😡  22: 😠  23: 🤯  24: 💔  25: 🥺  26: 😨  27: 😳\n> "
).strip()

try:
    emojis = " ".join(emoji_map[int(num)] for num in emoji_numbers.split())
except:
    print("❗ Invalid emoji number. Use only numbers from the emoji list.")
    exit()

engagement_str = input(
    "\n📊 On a scale from 0 to 100, how engaged did you feel while watching the video?\n"
    "Enter a number where 0 means not engaged at all, and 100 means fully engaged.\n> "
).strip()

try:
    engagement = float(engagement_str)
    if not (0 <= engagement <= 100):
        raise ValueError("Out of range")
except:
    print("❗ Invalid engagement value. It must be a number between 0 and 100.")
    exit()

# --- Process and predict
input_text = f"{comment} {emoji.demojize(emojis, delimiters=(' ', ' '))}"
embedding = get_roberta_embedding(input_text)
engagement_scaled = engagement / 100.0
features = np.hstack((embedding, [engagement_scaled])).reshape(1, -1)

prediction = model.predict(features)[0]
sentiment = label_encoder.inverse_transform([prediction])[0]

print(f"\n🧠 Predicted Sentiment: **{sentiment}**")
print("✅ Chatbot session finished.")

