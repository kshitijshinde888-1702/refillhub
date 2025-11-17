
# STREAMLIT DASHBOARD FIXED VERSION
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.title("ReFill Hub – ML Insights & Prediction Dashboard")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

st.subheader("Dataset Preview")
st.write(df.head())

target = "Refills Rate"

# Handle nulls
for col in df.columns:
    if df[col].dtype=="object":
        df[col]=df[col].fillna(df[col].mode()[0])
    else:
        df[col]=df[col].fillna(df[col].median())

# Encoding
X=df.drop(columns=[target])
y=df[target]

for col in X.columns:
    if X[col].dtype=='object':
        X[col]=LabelEncoder().fit_transform(X[col])

# Safety stratify
if y.nunique()<2:
    st.error("Label column has only ONE class. ML cannot run.")
    st.stop()

if y.value_counts().min()<2:
    st.warning("Stratify disabled due to low class count.")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
else:
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

# Marketing Insight Chart
st.subheader("Top Marketing Insight: Age Distribution")
fig,ax=plt.subplots()
df["Age"].value_counts().plot(kind="bar",ax=ax)
st.pyplot(fig)

# ML Models
models={
"Decision Tree":DecisionTreeClassifier(),
"Random Forest":RandomForestClassifier(),
"GBRT":GradientBoostingClassifier()
}

results={}
plt.figure(figsize=(7,5))

for name,model in models.items():
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    prob=model.predict_proba(X_test)[:,1]

    results[name]=[
        accuracy_score(y_test,preds),
        precision_score(y_test,preds),
        recall_score(y_test,preds),
        f1_score(y_test,preds)
    ]

    fpr,tpr,_=roc_curve(y_test,prob)
    plt.plot(fpr,tpr,label=name)

plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curve")
st.pyplot(plt)

st.subheader("Model Performance Table")
res_df=pd.DataFrame(results,index=["Accuracy","Precision","Recall","F1-score"]).T
st.write(res_df)

# Prediction Section
st.header("Predict New Customer Refill Adoption")
with st.form("predict_form"):
    age=st.number_input("Age",18,80)
    income=st.number_input("Income",0)
    eco=st.number_input("Eco Awareness (1-5)",1,5)
    submitted=st.form_submit_button("Predict")

if submitted:
    new_df=pd.DataFrame([[age,income,eco]],columns=["Age","Income","Eco_Awareness"])
    model=RandomForestClassifier().fit(X,y)
    pred=model.predict(new_df)[0]
    st.subheader("Prediction Result")
    st.write("Refill Likely" if pred==1 else "Refill Unlikely")
