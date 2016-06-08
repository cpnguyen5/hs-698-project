--update empty string ('') values to NULL for each column in TABLE puf
UPDATE puf SET npi=NULL WHERE npi='';
UPDATE puf SET provider_last_name=NULL WHERE provider_last_name='';
UPDATE puf SET provider_first_name=NULL WHERE provider_first_name='';
UPDATE puf SET provider_middle_initial=NULL WHERE provider_middle_initial='';
UPDATE puf SET provider_credentials=NULL WHERE provider_credentials='';
UPDATE puf SET provider_gender=NULL WHERE provider_gender='';
UPDATE puf SET provider_entity_type=NULL WHERE provider_entity_type='';
UPDATE puf SET provider_street_address_1=NULL WHERE provider_street_address_1='';
UPDATE puf SET provider_street_address_2=NULL WHERE provider_street_address_2='';
UPDATE puf SET provider_city=NULL WHERE provider_city='';
UPDATE puf SET provider_zip_code=NULL WHERE provider_zip_code='';
UPDATE puf SET provider_state_code=NULL WHERE provider_state_code='';
UPDATE puf SET provider_country_code=NULL WHERE provider_country_code='';
UPDATE puf SET provider_type=NULL WHERE provider_type='';
UPDATE puf SET medicare_participation_indicator=NULL WHERE medicare_participation_indicator='';
UPDATE puf SET place_of_service=NULL WHERE place_of_service='';
UPDATE puf SET HCPCS_code=NULL WHERE HCPCS_code='';
UPDATE puf SET HCPCS_description=NULL WHERE HCPCS_description='';
UPDATE puf SET identifies_HCPCS_as_drug_included_in_the_ASP_drug_list=NULL WHERE identifies_HCPCS_as_drug_included_in_the_ASP_drug_list='';
UPDATE puf SET number_of_services=NULL WHERE number_of_services='';
UPDATE puf SET number_of_medicare_beneficiaries=NULL WHERE number_of_medicare_beneficiaries='';
UPDATE puf SET number_of_distinct_medicare_beneficiary_per_day_services=NULL WHERE number_of_distinct_medicare_beneficiary_per_day_services='';
UPDATE puf SET average_medicare_allowed_amount=NULL WHERE average_medicare_allowed_amount='';
UPDATE puf SET average_submitted_charge_amount=NULL WHERE average_submitted_charge_amount='';
UPDATE puf SET average_medicare_payment_amount=NULL WHERE average_medicare_payment_amount='';
UPDATE puf SET average_medicare_standardized_amount=NULL WHERE average_medicare_standardized_amount='';

--update empty string ('') values to NULL for each column in TABLE report
UPDATE report SET npi=NULL WHERE npi ='';
UPDATE report SET provider_last_name=NULL WHERE provider_last_name='';
UPDATE report SET provider_first_name=NULL WHERE provider_first_name='';
UPDATE report SET provider_middle_initial=NULL WHERE provider_middle_initial='';
UPDATE report SET provider_credentials=NULL WHERE provider_credentials='';
UPDATE report SET provider_gender=NULL WHERE provider_gender='';
UPDATE report SET provider_entity_type=NULL WHERE provider_entity_type='';
UPDATE report SET provider_street_address_1=NULL WHERE provider_street_address_1='';
UPDATE report SET provider_street_address_2=NULL WHERE provider_street_address_2='';
UPDATE report SET provider_city=NULL WHERE provider_city='';
UPDATE report SET provider_zip_code=NULL WHERE provider_zip_code='';
UPDATE report SET provider_state_code=NULL WHERE provider_state_code='';
UPDATE report SET provider_country_code=NULL WHERE provider_country_code='';
UPDATE report SET provider_type=NULL WHERE provider_type='';
UPDATE report SET medicare_participation_indicator=NULL WHERE medicare_participation_indicator='';
UPDATE report SET number_of_HCPCS=NULL WHERE number_of_HCPCS='';
UPDATE report SET number_of_services=NULL WHERE number_of_services='';
UPDATE report SET number_of_medicare_beneficiaries=NULL WHERE number_of_medicare_beneficiaries='';
UPDATE report SET total_submitted_charge_amount=NULL WHERE total_submitted_charge_amount='';
UPDATE report SET total_medicare_allowed_amount=NULL WHERE total_medicare_allowed_amount='';
UPDATE report SET total_medicare_payment_amount=NULL WHERE total_medicare_payment_amount='';
UPDATE report SET total_medicare_standardized_payment_amount=NULL WHERE total_medicare_standardized_payment_amount='';
UPDATE report SET drug_suppress_indicator=NULL WHERE drug_suppress_indicator='';
UPDATE report SET number_of_HCPCS_associated_with_drug_services=NULL WHERE number_of_HCPCS_associated_with_drug_services='';
UPDATE report SET number_of_drug_services=NULL WHERE number_of_drug_services='';
UPDATE report SET number_of_medicare_beneficiaries_with_drug_services=NULL WHERE number_of_medicare_beneficiaries_with_drug_services='';
UPDATE report SET total_drug_submitted_charge_amount=NULL WHERE total_drug_submitted_charge_amount='';
UPDATE report SET total_drug_medicare_allowed_amount=NULL WHERE total_drug_medicare_allowed_amount='';
UPDATE report SET total_drug_medicare_payment_amount=NULL WHERE total_drug_medicare_payment_amount='';
UPDATE report SET total_drug_medicare_standardized_payment_amount=NULL WHERE total_drug_medicare_standardized_payment_amount='';
UPDATE report SET medical_suppress_indicator=NULL WHERE medical_suppress_indicator='';
UPDATE report SET number_of_HCPCS_associated_medical_services=NULL WHERE number_of_HCPCS_associated_medical_services='';
UPDATE report SET number_of_medical_services=NULL WHERE number_of_medical_services='';
UPDATE report SET number_of_medicare_beneficiaries_with_medical_services=NULL WHERE number_of_medicare_beneficiaries_with_medical_services='';
UPDATE report SET total_medical_submitted_charge_amount=NULL WHERE total_medical_submitted_charge_amount='';
UPDATE report SET total_medical_medicare_allowed_amount=NULL WHERE total_medical_medicare_allowed_amount='';
UPDATE report SET total_medical_medicare_payment_amount=NULL WHERE total_medical_medicare_payment_amount='';
UPDATE report SET total_medical_medicare_standardized_payment_amount=NULL WHERE total_medical_medicare_standardized_payment_amount='';
UPDATE report SET average_age_of_beneficiaries=NULL WHERE average_age_of_beneficiaries='';
UPDATE report SET number_of_beneficiaries_age_less_65=NULL WHERE number_of_beneficiaries_age_less_65='';
UPDATE report SET number_of_beneficiaries_age_65_to_74=NULL WHERE number_of_beneficiaries_age_65_to_74='';
UPDATE report SET number_of_beneficiaries_age_75_to_84=NULL WHERE number_of_beneficiaries_age_75_to_84='';
UPDATE report SET number_of_beneficiaries_age_greater_84=NULL WHERE number_of_beneficiaries_age_greater_84='';
UPDATE report SET number_of_female_beneficiaries=NULL WHERE number_of_female_beneficiaries='';
UPDATE report SET number_of_male_beneficiaries=NULL WHERE number_of_male_beneficiaries='';
UPDATE report SET number_of_non_hispanic_white_beneficiaries=NULL WHERE number_of_non_hispanic_white_beneficiaries='';
UPDATE report SET number_of_african_american_beneficiaries=NULL WHERE number_of_african_american_beneficiaries='';
UPDATE report SET number_of_asian_pacific_islander_beneficiaries=NULL WHERE number_of_asian_pacific_islander_beneficiaries='';
UPDATE report SET number_of_hispanic_beneficiaries=NULL WHERE number_of_hispanic_beneficiaries='';
UPDATE report SET number_of_american_indian_alaskan_native_beneficiaries=NULL WHERE number_of_american_indian_alaskan_native_beneficiaries='';
UPDATE report SET number_of_beneficiaries_with_race_not_elsewhere_classified=NULL WHERE number_of_beneficiaries_with_race_not_elsewhere_classified='';
UPDATE report SET number_of_beneficiaries_with_medicare_only_entitlement=NULL WHERE number_of_beneficiaries_with_medicare_only_entitlement='';
UPDATE report SET number_of_beneficiaries_with_medicare_and_medicaid_entitlement=NULL WHERE number_of_beneficiaries_with_medicare_and_medicaid_entitlement='';
UPDATE report SET percent_of_beneficiaries_identified_with_atrial_fibrillation=NULL WHERE percent_of_beneficiaries_identified_with_atrial_fibrillation='';
UPDATE report SET percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia=NULL WHERE percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia='';
UPDATE report SET percent_of_beneficiaries_identified_with_asthma=NULL WHERE percent_of_beneficiaries_identified_with_asthma='';
UPDATE report SET percent_of_beneficiaries_identified_with_cancer=NULL WHERE percent_of_beneficiaries_identified_with_cancer='';
UPDATE report SET percent_of_beneficiaries_identified_with_heart_failure=NULL WHERE percent_of_beneficiaries_identified_with_heart_failure='';
UPDATE report SET percent_of_beneficiaries_identified_with_chronic_kidney_disease=NULL WHERE percent_of_beneficiaries_identified_with_chronic_kidney_disease='';
UPDATE report SET percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease=NULL WHERE percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease='';
UPDATE report SET percent_of_beneficiaries_identified_with_depression=NULL WHERE percent_of_beneficiaries_identified_with_depression='';
UPDATE report SET percent_of_beneficiaries_identified_with_diabetes=NULL WHERE percent_of_beneficiaries_identified_with_diabetes='';
UPDATE report SET percent_of_beneficiaries_identified_with_hyperlipidemia=NULL WHERE percent_of_beneficiaries_identified_with_hyperlipidemia='';
UPDATE report SET percent_of_beneficiaries_identified_with_hypertension=NULL WHERE percent_of_beneficiaries_identified_with_hypertension='';
UPDATE report SET percent_of_beneficiaries_identified_with_ischemic_heart_disease=NULL WHERE percent_of_beneficiaries_identified_with_ischemic_heart_disease='';
UPDATE report SET percent_of_beneficiaries_identified_with_osteoporosis=NULL WHERE percent_of_beneficiaries_identified_with_osteoporosis='';
UPDATE report SET percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis=NULL WHERE percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis='';
UPDATE report SET percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders=NULL WHERE percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders='';
UPDATE report SET percent_of_beneficiaries_identified_with_stroke=NULL WHERE percent_of_beneficiaries_identified_with_stroke='';
UPDATE report SET average_HCC_risk_score_of_beneficiaries=NULL WHERE average_HCC_risk_score_of_beneficiaries='';

