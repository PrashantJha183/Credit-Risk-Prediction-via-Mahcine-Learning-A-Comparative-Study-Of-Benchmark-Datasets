import pandas as pd

# German dataset columns
column_names = [
    "Status_Checking_Account",
    "Duration",
    "Credit_History",
    "Purpose",
    "Credit_Amount",
    "Savings_Account",
    "Employment_Since",
    "Installment_Rate",
    "Personal_Status_Sex",
    "Other_Debtors",
    "Present_Residence",
    "Property",
    "Age",
    "Other_Installment_Plans",
    "Housing",
    "Number_Credits",
    "Job",
    "Number_People_Maintained",
    "Telephone",
    "Foreign_Worker",
    "Credit_Risk"
]

# Read german.data
df = pd.read_csv(
    "data/german.data",
    sep=' ',
    header=None,
    names=column_names
)

# Save as CSV
df.to_csv("data/south_german_credit.csv", index=False)
print("Conversion complete! Saved as data/south_german_credit.csv")
