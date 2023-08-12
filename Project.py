import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from matplotlib import pyplot as plt
import io

st.title("Data-PreProcessing Web App")
st.markdown("---")
st.markdown("""
<style>
.css-nqowgj.edgvbvh3
{
visibility:hidden;
}
.css-h5rgaw.egzxvld1
{
visibility:hidden;
}
</style>
""",unsafe_allow_html=True)



# Wrap the code in a try-except block
try:
    st.sidebar.subheader("Data Import")
    sel_file = st.sidebar.radio("Chose a file type", options=(".Xlsx", ".Csv"))
    if sel_file == ".Xlsx":
        # Create a file uploader widget
        file = st.sidebar.file_uploader('Upload an XLSX file', type=['xlsx'])
        # Check if a file was uploaded

        if file is not None:
            # Read the file as a pandas DataFrame
            df = pd.read_excel(file)
            # Display the DataFrame
            st.write('### Original Data:')
            st.write(df)
        else:
            st.write('Please upload an XLSX file.')
    elif sel_file == ".Csv":
        # Create a file uploader widget
        file = st.sidebar.file_uploader('Upload a CSV file', type=['csv'])
        # Check if a file was uploaded
        if file is not None:
            # Read the file as a pandas DataFrame
            df = pd.read_csv(file)
            # Display the DataFrame
            st.write('### Original Data:')
            st.write(df)
        else:
            st.write('Please upload a CSV file.')

    st.sidebar.markdown("---")

    st.markdown("---")
    st.write("Description:",df.describe(include='all'))
    st.markdown("---")

    # Handle_missing_values
    def handle_missing_values(df):
        # Count the number of missing values per column
        missing_values = df.isnull().sum()

        # Filter out columns with no missing values
        missing_values = missing_values[missing_values > 0]

        # Calculate the percentage of missing values per column
        missing_percent = missing_values / df.shape[0] * 100

        # Print out the columns with missing values and their percentage
        st.write('### Columns with missing values & Percentage:')
        st.write(missing_percent)

        # Find the total missing value percentage
        total_missing = df.isnull().sum().sum()
        total_cells = df.size
        missing_percentage = round((total_missing / total_cells) * 100, 2)
        # Display the result
        st.write(f"Total missing value percentage: {missing_percentage}%")

        # Select the method to handle missing values
        missing = st.sidebar.radio('Select a method to handle missing values:',
                                   options=("Drop missing values", "Fill missing values"))

        if missing == 'Drop missing values':
            # Drop rows with any missing values
            df = df.dropna()
            st.write('### Data after dropping rows with missing values:')
            st.write(df)
        elif missing == 'Fill missing values':
            # Select the column and method to fill missing values
            column = st.sidebar.multiselect('Select a column to fill missing values:', missing_values.index)
            method = st.sidebar.radio('Select a method to fill missing values:', options=("Mean", "Median", "Mode"))

            # Fill missing values with the selected method
            if method == 'Mean':
                df[column] = df[column].fillna(df[column].mean())
            elif method == 'Median':
                df[column] = df[column].fillna(df[column].median())
            elif method == 'Mode':
                # Fill missing values with mode
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])

            st.write('### Data after filling missing values:')
            st.write(df)
        return (df)


  # Define a function to handle outliers
    def handle_outliers(df):
        # Get the list of column names from the data frame
        column_names = list(df.columns)
        # Allow the user to select the columns to check for outliers
        selected_columns = st.multiselect('Select columns to check for outliers', column_names)
        # Define the threshold for outlier detection (default is 1.5)
        threshold = 1.5
        # Calculate the IQR and find the outliers for each selected column
        outliers = {}
        for column_name in selected_columns:
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[column_name] = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]

        # Calculate the percentage of outliers for each selected column
        total_rows = len(df)
        percent_outliers = {}
        for column_name, outlier_df in outliers.items():
            num_outliers = len(outlier_df)
            percent_outliers[column_name] = (num_outliers / total_rows) * 100

        # Display the results
        for column_name, percent in percent_outliers.items():
            st.write(f'Percentage of outliers in {column_name}: {percent:.2f}%')

        
        # Select the method to handle outliers
        option = st.sidebar.radio('Select a method to handle outliers:',
                                  ('Do nothing', 'Remove outliers', 'Replace outliers'))
        if option == 'Do nothing':
            # Do nothing to the DataFrame
            st.write('### Data without outlier treatment:')
            st.write(df)
        elif option == 'Remove outliers':
            # Select the column to remove outliers from
            column = st.selectbox('Select a column to remove outliers from:', df.columns)
            # Calculate the interquartile range (IQR) for the column
            q1, q3 = np.percentile(df[column].dropna(), [25, 75])
            iqr = q3 - q1
            # Calculate the lower and upper bounds for outliers
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            # Remove outliers outside the lower and upper bounds
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            # Display the DataFrame after removing outliers
            st.write('### Data after removing outliers:')
            st.write(df)
        elif option == 'Replace outliers':
            # Select the column to replace outliers in
            column = st.selectbox('Select a column to replace outliers in:', df.columns)
            # Calculate the interquartile range (IQR) for the column
            q1, q3 = np.percentile(df[column].dropna(), [25, 75])
            iqr = q3 - q1
            # Calculate the lower and upper bounds for outliers
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            # Replace outliers outside the lower and upper bounds with the median
            df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), df[column].median(),
                                  df[column])
            # Display the DataFrame after replacing outliers
            st.write('### Data after replacing outliers:')
            st.write(df)
        return (df)


    # Handle feature scaling
    def handle_scaling(df):
        feature_scale = st.sidebar.radio("Chose a Feature Scaling Method ", options=(
            "Standard Scalar", "Min Max Scalar", "Max Absolute Scalar", "Robust Scalar"))
        # Select the columns to scale
        columns = df.columns.tolist()
        selected_cols = st.multiselect("Select columns to scale:", df.columns)

        if len(columns) > 0:
            if feature_scale == 'Standard Scalar':
                if selected_cols:
                    scaler = StandardScaler()
                    df[selected_cols] = scaler.fit_transform(df[selected_cols])
            elif feature_scale == 'Min Max Scalar':
                if selected_cols:
                    scaler = MinMaxScaler()
                    df[selected_cols] = scaler.fit_transform(df[selected_cols])
            elif feature_scale == 'Max Absolute Scalar':
                if selected_cols:
                    scaler = MaxAbsScaler()
                    df[selected_cols] = scaler.fit_transform(df[selected_cols])
            elif feature_scale == 'Robust Scalar':
                if selected_cols:
                    scaler = RobustScaler()
                    df[selected_cols] = scaler.fit_transform(df[selected_cols])
            # Show the resulting dataset
            st.write(df)
        return (df)


    opt = st.sidebar.radio("What would you like to do ?",
                           options=("Missing Value Treatment", "Outlier Treatment", "Feature scaling"))
    if opt == "Missing Value Treatment":
        handle_missing_values(df)
    elif opt == "Outlier Treatment":
        handle_outliers(df)
    elif opt == "Feature scaling":
        handle_scaling(df)

    # Save the resulting data to a file
    save_option = st.sidebar.radio('Select a file format to save the data:', options=("xlsx", "csv"))
    if save_option == 'xlsx':
        
        # Create a file-like buffer to receive the output
        output_buffer = io.BytesIO()
        # Write the Excel data to the buffer
        df.to_excel(output_buffer, index=False)
        # Set up the download button to download the Excel file
        output_file = st.sidebar.download_button('Download Excel file', data=output_buffer.getvalue(),
                                                 file_name='output.xlsx', mime='application/vnd.ms-excel')

    elif save_option == 'csv':
        # Save the data to a csv file using pandas library
        output_file = st.sidebar.download_button('Download CSV file', df.to_csv(), file_name='output.csv',
                                                 mime='text/csv')




except:
    # Handle the error without displaying it
    pass



