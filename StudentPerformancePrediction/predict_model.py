import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import os
import joblib

# ====================
# Load and Prepare Data
# ====================
df = pd.read_csv("data/student_data.csv", sep=',')
df.columns = df.columns.str.strip()
print("Fixed columns:", df.columns.tolist())

# Create target label: Pass (1) or Fail (0)
df['Pass'] = df['Marks'].apply(lambda x: 1 if x >= 40 else 0)

# ==========================
# Regression: Predict Marks
# ==========================
X_reg = df[['Attendance', 'Study Hour', 'Test1', 'Test2', 'Internal']]
y_reg = df['Marks']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

df['Predicted_Marks'] = reg_model.predict(X_reg)

r2 = r2_score(y_reg, df['Predicted_Marks'])
mse = mean_squared_error(y_reg, df['Predicted_Marks'])

os.makedirs("output", exist_ok=True)
joblib.dump(reg_model, "output/student_marks_model.pkl")

# =====================================
# Classification: Predict Pass/Fail
# =====================================
X_clf = df[['Attendance', 'Study Hour', 'Test1', 'Test2', 'Internal']]
y_clf = df['Pass']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train_clf, y_train_clf)

pass_pred = clf_model.predict(X_clf)
df['Predicted_Pass'] = ['Pass ✅' if p == 1 else 'Fail ❌' for p in pass_pred]

acc = accuracy_score(y_test_clf, clf_model.predict(X_test_clf))
report = classification_report(y_test_clf, clf_model.predict(X_test_clf))

joblib.dump(clf_model, "output/student_pass_model.pkl")

df.to_csv("output/predicted_daa_output.csv", index=False)

# ====================================
# Single Student Prediction Function
# ====================================
def predict_single_student():
    print("\n📊 Enter new student data to predict Marks and Pass/Fail:")
    try:
        attendance = float(input("Attendance (%): "))
        study_hour = float(input("Study Hours per day: "))
        test1 = int(input("Test1 Score: "))
        test2 = int(input("Test2 Score: "))
        internal = int(input("Internal Score: "))

        new_data = pd.DataFrame({
            'Attendance': [attendance],
            'Study Hour': [study_hour],
            'Test1': [test1],
            'Test2': [test2],
            'Internal': [internal]
        })

        reg_model = joblib.load("output/student_marks_model.pkl")
        clf_model = joblib.load("output/student_pass_model.pkl")

        predicted_marks = reg_model.predict(new_data)[0]
        pass_status = clf_model.predict(new_data)[0]
        pass_prob = clf_model.predict_proba(new_data)[0][1]

        new_data['Predicted_Marks'] = predicted_marks
        new_data['Predicted_Pass'] = pass_status
        new_data['Pass_Probability'] = pass_prob

        os.makedirs("predictions", exist_ok=True)
        new_data.to_csv("predictions/single_student_prediction.csv", index=False)

        print(f"\n🎯 Predicted Marks: {predicted_marks:.2f}")
        print(f"✅ Predicted Status: {'Pass ✅' if pass_status == 1 else 'Fail ❌'}")
        print(f"📊 Probability of Passing: {pass_prob * 100:.2f}%")
        print("📁 Prediction saved to: predictions/single_student_prediction.csv")

    except Exception as e:
        print("⚠️ Error in prediction:", e)

# ====================================
# Batch File Prediction Function
# ====================================
def predict_from_file():
    try:
        file_path = input("📄 Enter path to CSV file with student data: ").strip()
        batch_df = pd.read_csv(file_path)
        required_cols = ['Attendance', 'Study Hour', 'Test1', 'Test2', 'Internal']

        if not all(col in batch_df.columns for col in required_cols):
            print(f"⚠️ Input file must contain columns: {required_cols}")
            return

        reg_model = joblib.load("output/student_marks_model.pkl")
        clf_model = joblib.load("output/student_pass_model.pkl")

        batch_df['Predicted_Marks'] = reg_model.predict(batch_df[required_cols])
        batch_df['Predicted_Pass'] = clf_model.predict(batch_df[required_cols])
        batch_df['Pass_Probability'] = clf_model.predict_proba(batch_df[required_cols])[:, 1]

        os.makedirs("predictions", exist_ok=True)
        output_path = "predictions/batch_student_predictions.csv"
        batch_df.to_csv(output_path, index=False)

        print(f"\n✅ Batch predictions completed.")
        print(f"📁 Results saved to: {output_path}")
        print("\n📊 Predicted Results:\n")
        print(batch_df.to_string(index=False))
        print("\n🎉 Batch prediction successful!")

    except Exception as e:
        print("⚠️ Error during batch prediction:", e)

# ===============================
# Choose Mode: Single or Batch
# ===============================
mode = input("\n🔍 Predict for (1) Single Student or (2) Multiple Students from File? (Enter 1 or 2): ").strip()
if mode == '1':
    predict_single_student()
elif mode == '2':
    predict_from_file()
else:
    print("⚠️ Invalid option. Please enter 1 or 2.")
