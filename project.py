import sqlite3
import os
import pandas as pd
import numpy as np
import csv

def get_path():
    f_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'api', 'dataset')
    return f_name


def readCSV():
    columns = ["npi", "provider_last_name", "provider_first_name", "provider_middle_initial", "provider_credentials",
               "provider_gender", "provider_entity_type", "provider_street_address_1", "provider_street_address_2",
               "provider_city", "provider_zip_code", "provider_state_code", "provider_country_code", "provider_type",
               "medicare_participation_indicator", "number_of_HCPCS", "number_of_services", "number_of_medicare_beneficiaries",
               "total_submitted_charge_amount", "total_medicare_allowed_amount", "total_medicare_payment_amount",
               "total_medicare_standardized_payment_amount", "drug_suppress_indicator",
               "number_of_HCPCS_associated_with_drug_services", "number_of_drug_services",
               "number_of_medicare_beneficiaries_with_drug_services", "total_drug_submitted_charge_amount",
               "total_drug_medicare_allowed_amount", "total_drug_medicare_payment_amount", "total_drug_medicare_standardized_payment_amount",
               "medical_suppress_indicator", "number_of_HCPCS_associated_medical_services", "number_of_medical_services",
               "number_of_medicare_beneficiaries_with_medical_services", "total_medical_submitted_charge_amount",
               "total_medical_medicare_allowed_amount", "total_medical_medicare_payment_amount",
               "total_medical_medicare_standardized_payment_amount", "average_age_of_beneficiaries", "number_of_beneficiaries_age_less_65",
               "number_of_beneficiaries_age_65_to_74", "number_of_beneficiaries_age_75_to_84", "number_of_beneficiaries_age_greater_84",
               "number_of_female_beneficiaries", "number_of_male_beneficiaries", "number_of_non_hispanic_white_beneficiaries",
               "number_of_african_american_beneficiaries", "number_of_asian_pacific_islander_beneficiaries",
               "number_of_hispanic_beneficiaries", "number_of_american_indian_alaskan_native_beneficiaries",
               "number_of_beneficiaries_with_race_not_elsewhere_classified", "number_of_beneficiaries_with_medicare_only_entitlement",
               "number_of_beneficiaries_with_medicare_and_medicaid_entitlement", "percent_of_beneficiaries_identified_with_atrial_fibrillation",
               "percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia",
               "percent_of_beneficiaries_identified_with_asthma", "percent_of_beneficiaries_identified_with_cancer",
               "percent_of_beneficiaries_identified_with_heart_failure", "percent_of_beneficiaries_identified_with_chronic_kidney_disease",
               "percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease", "percent_of_beneficiaries_identified_with_depression",
               "percent_of_beneficiaries_identified_with_diabetes", "percent_of_beneficiaries_identified_with_hyperlipidemia",
               "percent_of_beneficiaries_identified_with_hypertension", "percent_of_beneficiaries_identified_with_ischemic_heart_disease",
               "percent_of_beneficiaries_identified_with_osteoporosis", "percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis",
               "percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders",
               "percent_of_beneficiaries_identified_with_stroke", "average_HCC_risk_score_of_beneficiaries"]

    data_type = {"npi": int, "provider_last_name": str, "provider_first_name": str, "provider_middle_initial": str,
                 "provider_credentials": str,
                 "provider_gender": str, "provider_entity_type": str, "provider_street_address_1": str,
                 "provider_street_address_2": str, "provider_city": str,
                 "provider_zip_code": int, "provider_state_code": str, "provider_country_code": str, "provider_type": str,
                 "medicare_participation_indicator": str,
                 "number_of_HCPCS": int, "number_of_services": int, "number_of_medicare_beneficiaries": int,
                 "total_submitted_charge_amount": float,
                 "total_medicare_allowed_amount": float, "total_medicare_payment_amount": float,
                 "total_medicare_standardized_payment_amount": float,
                 "drug_suppress_indicator": str, "number_of_HCPCS_associated_with_drug_services": int,
                 "number_of_drug_services": int,
                 "number_of_medicare_beneficiaries_with_drug_services": int, "total_drug_submitted_charge_amount": float,
                 "total_drug_medicare_allowed_amount": float,
                 "total_drug_medicare_payment_amount": float, "total_drug_medicare_standardized_payment_amount": float,
                 "medical_suppress_indicator": str,
                 "number_of_HCPCS_associated_medical_services": int, "number_of_medical_services": int,
                 "number_of_medicare_beneficiaries_with_medical_services": int,
                 "total_medical_submitted_charge_amount": float, "total_medical_medicare_allowed_amount": float,
                 "total_medical_medicare_payment_amount": float,
                 "total_medical_medicare_standardized_payment_amount": float, "average_age_of_beneficiaries": int,
                 "number_of_beneficiaries_age_less_65": int,
                 "number_of_beneficiaries_age_65_to_74": int, "number_of_beneficiaries_age_75_to_84": int,
                 "number_of_beneficiaries_age_greater_84": int,
                 "number_of_female_beneficiaries": int, "number_of_male_beneficiaries": int,
                 "number_of_non_hispanic_white_beneficiaries": int,
                 "number_of_african_american_beneficiaries": int, "number_of_asian_pacific_islander_beneficiaries": int,
                 "number_of_hispanic_beneficiaries": int,
                 "number_of_american_indian_alaskan_native_beneficiaries": int,
                 "number_of_beneficiaries_with_race_not_elsewhere_classified": int,
                 "number_of_beneficiaries_with_medicare_only_entitlement": int,
                 "number_of_beneficiaries_with_medicare_and_medicaid_entitlement": int,
                 "percent_of_beneficiaries_identified_with_atrial_fibrillation": int,
                 "percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia": int,
                 "percent_of_beneficiaries_identified_with_asthma": int,
                 "percent_of_beneficiaries_identified_with_cancer": int,
                 "percent_of_beneficiaries_identified_with_heart_failure": int,
                 "percent_of_beneficiaries_identified_with_chronic_kidney_disease": int,
                 "percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease": int,
                 "percent_of_beneficiaries_identified_with_depression": int,
                 "percent_of_beneficiaries_identified_with_diabetes": int,
                 "percent_of_beneficiaries_identified_with_hyperlipidemia": int,
                 "percent_of_beneficiaries_identified_with_hypertension": int,
                 "percent_of_beneficiaries_identified_with_ischemic_heart_disease": int,
                 "percent_of_beneficiaries_identified_with_osteoporosis": int,
                 "percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis": int,
                 "percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders": int,
                 "percent_of_beneficiaries_identified_with_stroke": int,
                 "average_HCC_risk_score_of_beneficiaries": float}


    df = pd.read_csv(os.path.join(get_path(),
                                    'Medicare_Physician_and_Other_Supplier_National_Provider_Identifier__NPI__Aggregate_Report__Calendar_Year_2014.csv'),
                       sep=',', names=columns
                       # na_values=[''],
                       # dtype=data_type
                       )
    return df

def create_table():

    db_file = os.path.join(get_path(), 'cms.db')
    schema_file = os.path.join(get_path(), 'cms_schema.sql')
    update_file = os.path.join(get_path(), 'update.sql')

    db_is_new = not os.path.exists(db_file)
    #Connecting/Creating database file
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    if db_is_new:
        print "Database created, creating table(s) schema"
        print "Reading SQL script..."
        f_schema = open(schema_file, 'r')
        schema = f_schema.read()
        f_schema.close()
        c.executescript(schema) #create schema
        #import data into SQL db
        conn.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
        data = readCSV()
        cms = data.to_sql('report', con=conn, flavor='sqlite', if_exists='append', index=False)
        #Update empty strings into SQL NULL values
        f_update = open(update_file, 'r')
        update = f_update.read()
        f_update.close()
        c.executescript(update)

    else:
        print "Database exists; opened successfully"

    conn.commit()
    return "Database Initialization complete"


def query(path):

    db_file = os.path.join(path, 'cms.db')
    queries_file = os.path.join(path, "cms_queries.sql")
    print queries_file

    conn = sqlite3.connect(db_file) #create/open database
    print "Connected to database successfully"
    with conn:

        c = conn.cursor()

        print "Reading SQL script..."
        f_queries = open(queries_file, 'r')
        query = f_queries.read()
        f_queries.close()

        query_commands=query.split(';')
        # print query_commands

        print "Running SQL script..."
        query_lst=[]
        for command in query_commands:
            # try:
            c.execute(command)
            row = c.fetchall()
            query_lst+=[row]
            # except OperationalError, msg:
            #     print "Command skipped: ", msg


        # c.execute("SELECT provider_state_code, percent_of_beneficiaries_identified_with_cancer FROM report GROUP BY provider_state_code;")
        # print c.fetchone()
        # rows = c.fetchall()
        # for row in rows:
        #     print row
    # return rows
    return query_lst


#main
# data = readCSV()
# print data["provider_middle_initial"][1], data["provider_middle_initial"][1]
# print get_path()
# print create_table()
# print query(get_path())


# print data.isnull().sum()
# print data.dropna()
# print data.shape
# print data
# print data.dropna().dtypes
