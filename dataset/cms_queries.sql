SELECT percent_of_beneficiaries_identified_with_cancer 
    FROM report;

SELECT AVG(percent_of_beneficiaries_identified_with_cancer)
    FROM report 
    WHERE provider_state_code = 'CA';

SELECT provider_state_code, AVG(percent_of_beneficiaries_identified_with_cancer) 
    FROM report 
    GROUP BY provider_state_code;

SELECT provider_state_code, provider_city, AVG(percent_of_beneficiaries_identified_with_cancer) 
    FROM report 
    GROUP BY provider_state_code, provider_city WHERE provider_state_code = 'CA';

SELECT provider_state_code, SUM(number_of_services) 
    FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
    GROUP BY provider_state_code;

SELECT provider_type, SUM(number_of_services)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_type;

SELECT provider_state_code, AVG(total_submitted_charge_amount)
   FROM report
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(total_medicare_allowed_amount)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(total_medicare_payment_amount)
   FROM report
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(average_age_of_beneficiaries)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;


SELECT provider_state_code, AVG(number_of_female_beneficiaries)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(number_of_male_beneficiaries)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(number_of_non_hispanic_white_beneficiaries)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(number_of_african_american_beneficiaries)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(number_of_asian_pacific_islander_beneficiaries)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(number_of_hispanic_beneficiaries)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(number_of_beneficiaries_with_race_not_elsewhere_classified)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(number_of_beneficiaries_with_medicare_only_entitlement)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;

SELECT provider_state_code, AVG(number_of_beneficiaries_with_medicare_and_medicaid_entitlement)
   FROM report
   WHERE percent_of_beneficiaries_identified_with_cancer > 0
   GROUP BY provider_state_code;
a
