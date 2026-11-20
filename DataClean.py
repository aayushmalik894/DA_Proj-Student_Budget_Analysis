import pandas as pd
import numpy as np

def clean_student_spending(input_path: str, output_path: str) -> pd.DataFrame:
    # 1. Load dataset
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")

    # 2. Standardize column names (snake_case, stripped whitespace)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 3. Remove exact duplicates
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed: {initial_rows - len(df)}")

    # 4. Clean and standardize text columns
    cat_cols = ['gender', 'year_in_school', 'major', 'preferred_payment_method']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # 5. Handle missing values defensively (for future/updated data batches)
    num_cols = [
        'age', 'monthly_income', 'financial_aid', 'tuition', 'housing',
        'food', 'transportation', 'books_supplies', 'entertainment',
        'personal_care', 'technology', 'health_wellness', 'miscellaneous'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Impute numeric nulls with median if any exist
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        # Impute categorical nulls with mode if any exist
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 6. Data validation & outlier/anomaly checks
    # Ensure valid student age range and non-negative financial values
    df = df[(df['age'] >= 16) & (df['age'] <= 65)]
    financial_cols = [c for c in num_cols if c != 'age']
    for col in financial_cols:
        df = df[df[col] >= 0]

    # 7. Optimize data types for analytics & memory efficiency
    # Convert academic year into an ordered categorical type so charts sort chronologically
    year_order = ['Freshman', 'Sophomore', 'Junior', 'Senior']
    df['year_in_school'] = pd.Categorical(
        df['year_in_school'], categories=year_order, ordered=True
    )

    for col in ['gender', 'major', 'preferred_payment_method']:
        df[col] = df[col].astype('category')

    # 8. Feature Engineering for Analytics
    monthly_expense_cols = [
        'housing', 'food', 'transportation', 'books_supplies',
        'entertainment', 'personal_care', 'technology',
        'health_wellness', 'miscellaneous'
    ]
    
    # Total recurring monthly expenses (excluding lump-sum tuition)
    df['total_monthly_expenses'] = df[monthly_expense_cols].sum(axis=1)
    
    # Total monthly resources (income + aid)
    df['total_monthly_resources'] = df['monthly_income'] + df['financial_aid']
    
    # Monthly net cash flow (positive = surplus, negative = deficit)
    df['monthly_net_cash_flow'] = df['total_monthly_resources'] - df['total_monthly_expenses']
    
    # Expense-to-Income ratio (%)
    df['expense_to_resource_ratio'] = (
        (df['total_monthly_expenses'] / df['total_monthly_resources']) * 100
    ).round(2)

    # 9. Export cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Final shape: {df.shape}")
    print(f"Cleaned file saved successfully to: {output_path}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_student_spending(
        input_path="student_spending.csv",
        output_path="student_spending_cleaned.csv"
    )
    print("\nPreview of engineered features:")
    print(cleaned_df[['year_in_school', 'total_monthly_resources', 'total_monthly_expenses', 'monthly_net_cash_flow']].head())