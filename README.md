# telemarketing_bank
telemarketing_campaign_bank

The bank provides financial services/products such as savings accounts, current accounts, debit cards, etc. to its customers. In order to increase its overall revenue, the bank conducts various marketing campaigns for its financial products such as credit cards, term deposits, loans, etc. These campaigns are intended for the bank’s existing customers. However, the marketing campaigns need to be cost-efficient so that the bank not only increases their overall revenues but also the total profit. i have applied knowledge of EDA on the given dataset to analyse the patterns and provide inferences/solutions for the future marketing campaign.

The bank conducted a telemarketing campaign for one of its financial products ‘fixed Deposits’ to help foster long-term relationships with existing customers. The dataset contains information about all the customers who were contacted during a particular year to open fixed deposit accounts.

Term deposits also called fixed deposits, are the cash investments made for a specific time period ranging from 1 month to 5 years for predetermined fixed interest rates. The fixed interest rates offered for term deposits are higher than the regular interest rates for savings accounts. The customers receive the total amount (investment plus the interest) at the end of the maturity period. Also, the money can only be withdrawn at the end of the maturity period. Withdrawing money before that will result in an added penalty associated, and the customer will not receive any interest returns.

Following are the some of the types of variables:

Numeric data type: banking dataset: salary, balance, duration and age.
Categorical data type: banking dataset: education, job, marital, poutcome and month etc.
Ordinal data type: banking dataset: Age group.
Time and date type
Coordinates type of data: latitude and longitude type

# Since the data does not contain any missing values Imputation Processes are not required
# In case there are missing values in the data, then we can  remove those columns or features that have more than 25% of rows as missing, this is generally preferred.

Check if the features show a slope at all
#If they do, then do you see some deciles below the population average and some higher than population average?
#If that is the case then the slope will be strong
#Conclusion: A strong slope is indicative of the features' ability to discriminate the event from non event
#            making it a good predictor

##### Recommendations ##########
# Phase 1 - Speak to customers who falls in top 2 predicted ranks(deciles) and also are in High Duration segment
# Similarly - Phase 2 ( Top 2 , Low Duration)
# Phase 3 - Bottom 7 and High Duration (randomly select 25% of them only)

reference papers:A_data_driven_approach_to_predict_the_success_of_bank_telemarketing
